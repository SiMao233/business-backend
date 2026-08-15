"""系统角色管理 Repository 层：数据访问（CRUD / 查询）。"""

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.common.pagination import PageParams, PageResult, paginate
from app.common.repository import BaseRepository
from app.modules.system.permission.model import Permission
from app.modules.system.role.model import Role


class RoleRepository(BaseRepository):
    """后台角色数据访问仓库。"""

    # 初始化仓库：绑定数据库会话与 Role 模型
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, Role)

    # 根据主键 ID 查询角色（预加载权限关联），未找到时返回 None
    async def get_by_id(self, role_id: UUID) -> Role | None:
        stmt = (
            select(Role)
            .options(selectinload(Role.permissions))
            .where(Role.id == role_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 根据角色编码 code 查询角色，未找到时返回 None
    async def get_by_code(self, code: str) -> Role | None :
        stmt = select(Role).where(Role.code == code)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 分页查询角色列表，支持按名称或编码关键字模糊搜索
    async def list_page(
        self,
        params: PageParams,
        name: str | None = None,
        code: str | None = None,
        status: int | None = None,
    ) -> PageResult[Any]:
        stmt = (
            select(Role)
            .options(selectinload(Role.permissions))
            .order_by(Role.id.desc())
        )
        if name:
            like = f"%{name}%"
            stmt = stmt.where(Role.name.like(like))
        if code:
            stmt = stmt.where(Role.code == code)
        if status is not None:
            stmt = stmt.where(Role.status == status)
        return await paginate(self.db, stmt, params)

    # 新增角色：写入数据库并提交事务，重新查询后返回角色对象（含权限关联）
    async def create(self, role: Role) -> Role:
        self.db.add(role)
        await self.db.commit()
        return await self._reload(role.id)

    # 更新角色：提交数据库变更并重新查询，返回更新后的角色对象
    async def update(self, role: Role) -> Role:
        await self.db.commit()
        return await self._reload(role.id)

    # 按主键重新查询角色（预加载权限关联）；刚提交的记录必然存在
    async def _reload(self, role_id: UUID) -> Role:
        stmt = (
            select(Role)
            .options(selectinload(Role.permissions))
            .where(Role.id == role_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 按 ID 批量查询权限（分配校验用）
    async def get_permissions_by_ids(self, permission_ids: list[UUID]) -> list[Permission]:
        if not permission_ids:
            return []
        stmt = select(Permission).where(Permission.id.in_(permission_ids))
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # 全量权限列表（父链补全用）
    async def list_all_permissions(self) -> list[Permission]:
        stmt = select(Permission).order_by(Permission.type.asc(), Permission.name.asc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # 全量覆盖绑定角色权限并提交（替换现有绑定）
    async def set_permissions(self, role: Role, permissions: list[Permission]) -> None:
        role.permissions = permissions
        await self.db.commit()

    # 删除角色：从数据库移除并提交事务（角色-权限关联由外键 CASCADE 清理）
    async def delete(self, role: Role) -> None:
        await self.db.delete(role)
        await self.db.commit()
