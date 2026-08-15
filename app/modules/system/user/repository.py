"""系统用户管理 Repository 层：数据访问（CRUD / 查询）。"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.common.pagination import PageParams, PageResult, paginate
from app.common.repository import BaseRepository
from app.modules.system.role.model import Role
from app.modules.system.user.model import User
from app.modules.system.user.schema import UserOut


class UserRepository(BaseRepository):
    """后台用户数据访问仓库。"""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, User)

    # 根据主键 ID 查询用户（预加载角色关系，避免异步懒加载报错）
    async def get_by_id(self, user_id: UUID) -> User | None:
        stmt = (
            select(User)
            .options(selectinload(User.roles))
            .where(User.id == user_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 根据用户名查询用户（唯一性校验 / 登录）
    async def get_by_username(self, username: str) -> User | None:
        stmt = select(User).where(User.username == username)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 根据邮箱查询用户（唯一性校验，可排除自身）
    async def get_by_email(self, email: str, exclude_id: UUID | None = None) -> User | None:
        stmt = select(User).where(User.email == email)
        if exclude_id is not None:
            stmt = stmt.where(User.id != exclude_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 按 ID 批量查询角色（用于给用户分配角色）
    async def get_roles_by_ids(self, role_ids: list[UUID]) -> list[Role]:
        if not role_ids:
            return []
        stmt = select(Role).where(Role.id.in_(role_ids))
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # 分页查询用户列表，支持关键字/状态/角色筛选
    async def list_page(
        self,
        params: PageParams,
        username: str | None = None,
        status: int | None = None,
        role_id: UUID | None = None,
    ) -> PageResult[UserOut]:
        stmt = (
            select(User)
            .options(selectinload(User.roles))
            .order_by(User.id.desc())
        )
        if username:
            like = f"%{username}%"
            stmt = stmt.where(
                User.username.like(like)
            )
        if status is not None:
            stmt = stmt.where(User.status == status)
        if role_id is not None:
            stmt = stmt.join(User.roles).where(Role.id == role_id)
        return await paginate(self.db, stmt, params)

    # 新增用户：写入数据库并提交事务，刷新后返回用户对象
    async def create(self, user: User) -> User:
        self.db.add(user)
        await self.db.commit()
        stmt = (
            select(User)
            .options(selectinload(User.roles))
            .where(User.id == user.id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 更新用户：提交数据库变更并刷新，返回更新后的用户对象
    async def update(self, user: User) -> User:
        await self.db.commit()
        stmt = (
            select(User)
            .options(selectinload(User.roles))
            .where(User.id == user.id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 删除用户：从数据库移除并提交事务
    async def delete(self, user: User) -> None:
        await self.db.delete(user)
        await self.db.commit()
