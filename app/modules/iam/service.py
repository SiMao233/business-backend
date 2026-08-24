"""IAM Service 层：认证业务逻辑（登录 / 刷新 / 登出 / 当前用户）。"""

import time
from uuid import UUID

import jwt
from loguru import logger
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.exceptions import NotFoundError, UnauthorizedError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from app.middleware.authentication import TOKEN_VERSION_PREFIX, UserContext
from app.modules.iam.repository import IamRepository
from app.modules.iam.schema import LoginRequest, RefreshRequest, TokenOut, UserInfoOut
from app.modules.system.user.model import User

settings = get_settings()


async def revoke_all_sessions(redis: Redis, user_id: UUID) -> None:
    """强制使某用户所有已签发令牌失效（重新登录 / 改密码 / 禁用账号时调用）。

    原理：递增该用户令牌版本号，之前签发令牌携带的旧版本号将无法通过校验。
    """
    await redis.incr(f"{TOKEN_VERSION_PREFIX}:{user_id.hex}")


class IamService:
    """用户身份与访问管理服务（认证相关）。"""

    # 刷新令牌 Redis 键前缀：iam:refresh:{user_id}:{jti}
    REFRESH_KEY_PREFIX = "iam:refresh"
    # access 令牌黑名单键前缀：iam:blacklist:{jti}
    ACCESS_BLACKLIST_PREFIX = "iam:blacklist"

    def __init__(self, db: AsyncSession, redis: Redis) -> None:
        self.repo = IamRepository(db)
        self.redis = redis

    # 登录：校验用户名密码与账号状态，签发令牌对并记录登录 IP
    async def login(self, req: LoginRequest, ip: str | None = None) -> TokenOut:
        user = await self.repo.get_by_username(req.username)
        # 用户不存在与密码错误返回相同提示，避免账号枚举
        if not user or not verify_password(req.password, user.password_hash):
            logger.warning("登录失败 username={} ip={}", req.username, ip)
            raise UnauthorizedError("用户名或密码错误")
        if user.status != 1:
            logger.warning("登录失败(账号禁用) username={} ip={}", req.username, ip)
            raise UnauthorizedError("账号已被禁用")
        await self.repo.update_login_info(user, ip or "")
        logger.info("用户登录成功 user_id={} ip={}", user.id.hex, ip)
        # 单会话：重新登录即递增版本号，使该用户之前所有令牌立即失效 （想要单会话就取消下面一行的注释）
        # await revoke_all_sessions(self.redis, user.id)
        return await self._issue_tokens(user)

    # 刷新令牌：校验旧 refresh（Redis 有状态）后轮换，旧令牌立即作废
    async def refresh(self, req: RefreshRequest) -> TokenOut:
        payload = self._decode_refresh(req.refresh_token)
        user_id = UUID(payload["sub"])
        # 版本号校验：被强制下线后旧 refresh 立即失效
        current_ver = await self._get_token_version(user_id)
        if payload.get("ver", 0) != current_ver:
            raise UnauthorizedError("刷新令牌已失效")
        key = self._refresh_key(payload)
        if await self.redis.get(key) != req.refresh_token:
            raise UnauthorizedError("刷新令牌已失效")
        user = await self.repo.get_by_id(user_id)
        if not user or user.status != 1:
            raise UnauthorizedError("用户不存在或已被禁用")
        await self.redis.delete(key)
        return await self._issue_tokens(user)

    # 登出：撤销 refresh（删除 Redis 记录）并拉黑当前 access（剩余有效期作 TTL），幂等
    async def logout(self, user: UserContext, req: RefreshRequest) -> None:
        try:
            payload = decode_token(req.refresh_token)
        except jwt.PyJWTError:
            pass
        else:
            if payload.get("token_type") == "refresh":
                await self.redis.delete(self._refresh_key(payload))
        if user.jti and user.exp:
            remaining = user.exp - int(time.time())
            if remaining > 0:
                await self.redis.setex(
                    f"{self.ACCESS_BLACKLIST_PREFIX}:{user.jti}", remaining, "1"
                )

    # 当前用户信息
    async def me(self, user_id: UUID) -> UserInfoOut:
        user = await self.repo.get_by_id(user_id)
        if not user:
            raise NotFoundError("用户不存在")
        return UserInfoOut.model_validate(user)

    # 读取当前令牌版本号（无记录视为 0）
    async def _get_token_version(self, user_id: UUID) -> int:
        val = await self.redis.get(f"{TOKEN_VERSION_PREFIX}:{user_id.hex}")
        return int(val) if val else 0

    # 强制踢掉用户所有会话（改密码 / 禁用账号时调用）
    async def revoke_all_sessions(self, user_id: UUID) -> None:
        await revoke_all_sessions(self.redis, user_id)

    # 签发令牌对：refresh token 写入 Redis（TTL = 配置的刷新有效期）
    async def _issue_tokens(self, user: User) -> TokenOut:
        version = await self._get_token_version(user.id)
        access_token = create_access_token(
            str(user.id.hex), version=version, username=user.username
        )
        refresh_token = create_refresh_token(str(user.id.hex), version=version)
        payload = decode_token(refresh_token)
        ttl = settings.jwt_refresh_token_expire_days * 86400
        await self.redis.setex(self._refresh_key(payload), ttl, refresh_token)
        return TokenOut(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.jwt_access_token_expire_minutes * 60,
            user=UserInfoOut.model_validate(user),
        )

    # 解析 refresh 令牌；失败 / 类型不符抛出 UnauthorizedError
    @staticmethod
    def _decode_refresh(token: str) -> dict[str, object]:
        try:
            payload = decode_token(token)
        except jwt.PyJWTError:
            raise UnauthorizedError("刷新令牌无效或已过期") from None
        if payload.get("token_type") != "refresh":
            raise UnauthorizedError("令牌类型错误")
        return payload

    # 构造刷新令牌 Redis 键
    @staticmethod
    def _refresh_key(payload: dict[str, object]) -> str:
        return f"{IamService.REFRESH_KEY_PREFIX}:{payload['sub']}:{payload['jti']}"
