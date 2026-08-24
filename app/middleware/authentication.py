"""认证中间件与当前用户依赖。

提供 `UserContext` 结构、中间件骨架与 `get_current_user` 依赖注入点。
`get_current_user` 已接入真实 JWT 校验：解析 Authorization 头、校验 access 令牌后
返回当前用户上下文；失败抛出 `UnauthorizedError`。
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

import jwt
from fastapi import Depends, Request
from redis.asyncio import Redis
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.exceptions import UnauthorizedError
from app.core.redis import get_redis
from app.core.security import decode_token

# access 令牌黑名单 Redis 键前缀：iam:blacklist:{jti}
ACCESS_BLACKLIST_PREFIX = "iam:blacklist"
# 每用户令牌版本号键前缀：iam:token_version:{user_id}（改密码/禁用/重登后全踢）
TOKEN_VERSION_PREFIX = "iam:token_version"


@dataclass(frozen=True)
class UserContext:
    """当前登录用户上下文。"""

    user_id: UUID | None = None
    username: str | None = None
    jti: str | None = None   # 当前 access 令牌的 jti（登出时用于拉黑）
    exp: int | None = None   # 当前 access 令牌过期时间戳（黑名单 TTL 依据）


class AuthenticationMiddleware(BaseHTTPMiddleware):  # pragma: no cover - 未启用（改用 get_current_user 依赖）
    """HTTP 认证中间件骨架（预留）。

    当前直接放行请求；若后续需要全局强制鉴权，可解析 Authorization 头校验 JWT 后
    将用户写入 `request.state.user`。业务接口鉴权请使用 `CurrentUserDep` 依赖。
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request.state.user = UserContext()
        return await call_next(request)


async def get_current_user(
    request: Request,
    redis: Annotated[Redis, Depends(get_redis)],
) -> UserContext:
    """FastAPI 依赖：解析 Authorization 头并校验 access 令牌，返回当前登录用户。

    除 JWT 签名/过期校验外，还会检查该令牌 jti 是否已进入黑名单（登出后立即失效）。
    """
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise UnauthorizedError("未提供认证令牌")
    token = auth.split(" ", 1)[1].strip()
    if not token:
        raise UnauthorizedError("未提供认证令牌")
    try:
        payload = decode_token(token)
    except jwt.PyJWTError:
        raise UnauthorizedError("认证令牌无效或已过期") from None
    if payload.get("token_type") != "access":
        raise UnauthorizedError("令牌类型错误")
    jti = payload["jti"]
    if await redis.get(f"{ACCESS_BLACKLIST_PREFIX}:{jti}"):
        raise UnauthorizedError("认证令牌已失效")
    # 版本号校验：改密码/禁用账号/重新登录后，该用户旧令牌全部失效
    current_ver = await redis.get(f"{TOKEN_VERSION_PREFIX}:{UUID(payload['sub']).hex}")
    if payload.get("ver", 0) != (int(current_ver) if current_ver else 0):
        raise UnauthorizedError("认证令牌已失效")
    user_ctx = UserContext(
        user_id=UUID(payload["sub"]),
        username=payload.get("username"),  # 旧令牌可能无该字段，取不到则为 None
        jti=jti,
        exp=int(payload["exp"]),
    )
    request.state.user = user_ctx
    return user_ctx


# 可直接用于业务接口声明的类型别名
CurrentUserDep = Annotated[UserContext, Depends(get_current_user)]
