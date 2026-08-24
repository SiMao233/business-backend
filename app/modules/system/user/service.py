"""系统用户管理 Service 层：业务规则、权限判断、流程编排。"""

from uuid import UUID

from loguru import logger
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageParams, PageResult
from app.core.exceptions import BizError, NotFoundError
from app.core.security import hash_password
from app.middleware.authentication import UserContext
from app.modules.iam.service import revoke_all_sessions
from app.modules.system.user.model import User
from app.modules.system.user.repository import UserRepository
from app.modules.system.user.schema import (
    UserCreate,
    UserOut,
    UserQuery,
    UserResetPassword,
    UserUpdate,
)


class UserService:
    """后台用户管理服务。"""

    # 初始化服务：创建用户数据仓库实例（绑定数据库会话与 Redis）
    def __init__(self, db: AsyncSession, redis: Redis) -> None:
        self.repo = UserRepository(db)
        self.redis = redis

    # 分页查询用户列表，支持按用户名 / 状态 / 角色 ID 筛选
    async def list_page(self, query: UserQuery) -> PageResult[UserOut]:
        page_params = PageParams(page=query.page, pageSize=query.pageSize)
        page = await self.repo.list_page(
            page_params,
            username=query.username,
            status=query.status,
            role_id=query.role_id,
        )
        return PageResult(
            list=[UserOut.model_validate(item) for item in page.list],
            total=page.total,
            page=page.page,
            pageSize=page.pageSize,
        )

    # 查询单个用户详情，用户不存在时抛出 NotFoundError
    async def get(self, user_id: UUID) -> UserOut:
        user = await self.repo.get_by_id(user_id)
        if not user:
            raise NotFoundError("用户不存在")
        return UserOut.model_validate(user)

    # 创建用户：校验用户名/邮箱唯一性与角色 ID 有效性，加密密码后入库
    async def create(self, req: UserCreate) -> UserOut:
        if await self.repo.get_by_username(req.username):
            raise BizError("用户名已存在")
        if req.email and await self.repo.get_by_email(req.email):
            raise BizError("邮箱已被使用")
        roles = await self.repo.get_roles_by_ids(req.role_ids)
        if len(roles) != len(set(req.role_ids)):
            raise BizError("存在无效的角色 ID")
        user = User(
            username=req.username,
            password_hash=hash_password(req.password),
            nickname=req.nickname,
            email=req.email,
            phone=req.phone,
            avatar=req.avatar,
            status=req.status,
            is_superuser=req.is_superuser,
        )
        user.roles = roles
        return UserOut.model_validate(await self.repo.create(user))

    # 更新用户：校验邮箱唯一性，支持更新角色关联
    async def update(self, user_id: UUID, req: UserUpdate, operator: UserContext | None = None) -> UserOut:
        user = await self.repo.get_by_id(user_id)
        if not user:
            raise NotFoundError("用户不存在")
        if req.email and req.email != user.email:
            if await self.repo.get_by_email(req.email, exclude_id=user_id):
                raise BizError("邮箱已被使用")
        user.nickname = req.nickname
        user.email = req.email
        user.phone = req.phone
        user.avatar = req.avatar
        user.status = req.status
        user.is_superuser = req.is_superuser
        if req.role_ids is not None:
            roles = await self.repo.get_roles_by_ids(req.role_ids)
            if len(roles) != len(set(req.role_ids)):
                raise BizError("存在无效的角色 ID")
            user.roles = roles
        result = UserOut.model_validate(await self.repo.update(user))
        # 禁用账号：强制踢掉该用户所有在线会话
        if req.status == 0:
            await revoke_all_sessions(self.redis, user_id)
        logger.info("更新用户 user_id={} by={}", user_id, operator.username if operator else "system")
        return result

    # 重置用户密码：重新生成并保存密码哈希（user_id 从请求体获取）
    async def reset_password(self, req: UserResetPassword) -> None:
        user = await self.repo.get_by_id(req.user_id)
        if not user:
            raise NotFoundError("用户不存在")
        user.password_hash = hash_password(req.password)
        await self.repo.update(user)
        # 改密码后强制踢掉该用户所有在线会话（旧密码对应的 token 全部失效）
        await revoke_all_sessions(self.redis, req.user_id)

    # 删除用户：超级管理员禁止删除；返回被删用户供审计记录
    async def delete(self, user_id: UUID, operator: UserContext | None = None) -> User:
        user = await self.repo.get_by_id(user_id)
        if not user:
            raise NotFoundError("用户不存在")
        if user.is_superuser:
            raise BizError("超级管理员不允许删除")
        await self.repo.delete(user)
        # 删除账号后强制踢掉其所有在线会话
        await revoke_all_sessions(self.redis, user_id)
        logger.warning(
            "删除用户 user_id={} username={} by={}",
            user_id, user.username, operator.username if operator else "system",
        )
        return user
