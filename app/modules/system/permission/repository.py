"""系统权限管理 Repository 层：数据访问（CRUD / 查询）。"""

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.common.pagination import PageParams, PageResult, paginate
from app.common.repository import BaseRepository
from app.modules.iam.model import RolePermission, UserRole
from app.modules.system.permission.model import Permission
from app.modules.system.role.model import Role
from app.modules.system.user.model import User


class PermissionRepository(BaseRepository):
    """后台权限数据访问仓库。"""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, Permission)

    # 根据主键 ID 查询权限，未找到返回 None
    async def get_by_id(self, permission_id: UUID) -> Permission | None:
        stmt = select(Permission).where(Permission.id == permission_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 根据权限码查询（唯一性校验，可排除自身）
    async def get_by_code(self, code: str, exclude_id: UUID | None = None) -> Permission | None:
        stmt = select(Permission).where(Permission.code == code)
        if exclude_id is not None:
            stmt = stmt.where(Permission.id != exclude_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 按 ID 批量查询权限（分配校验用）
    async def get_by_ids(self, permission_ids: list[UUID]) -> list[Permission]:
        if not permission_ids:
            return []
        stmt = select(Permission).where(Permission.id.in_(permission_ids))
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # 全量权限列表（树组装 / 父链补全 / 防环校验用）
    # 显式 selectinload children：自引用关系 lazy=selectin 不生效，需在此预加载避免异步懒加载
    async def list_all(self) -> list[Permission]:
        stmt = (
            select(Permission)
            .options(selectinload(Permission.children))
            .order_by(Permission.type.asc(), Permission.name.asc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # 分页查询权限列表，支持关键字 / 类型 / 状态筛选
    async def list_page(
        self,
        params: PageParams,
        keyword: str | None = None,
        type_: int | None = None,
        status: int | None = None,
    ) -> PageResult[Any]:
        stmt = select(Permission).order_by(Permission.type.asc(), Permission.id.desc())
        if keyword:
            like = f"%{keyword}%"
            stmt = stmt.where(Permission.name.like(like) | Permission.code.like(like))
        if type_ is not None:
            stmt = stmt.where(Permission.type == type_)
        if status is not None:
            stmt = stmt.where(Permission.status == status)
        return await paginate(self.db, stmt, params)

    # 根据用户 ID 聚合其所有启用角色绑定的权限 ID（去重，mine 接口用）
    async def get_permission_ids_by_user(self, user_id: UUID) -> set[UUID]:
        stmt = (
            select(RolePermission.permission_id)
            .join(Role, Role.id == RolePermission.role_id)
            .join(UserRole, UserRole.role_id == Role.id)
            .join(User, User.id == UserRole.user_id)
            .where(User.id == user_id)
            .where(Role.status == 1)
        )
        result = await self.db.execute(stmt)
        return set(result.scalars().all())

    # 判断用户是否为超级管理员（接口鉴权放行用）
    async def is_superuser(self, user_id: UUID) -> bool:
        stmt = select(User.is_superuser).where(User.id == user_id)
        result = await self.db.execute(stmt)
        return bool(result.scalar_one_or_none())

    # 对权限 ID 集合补全其所有祖先（父链），确保权限继承 / 树结构完整
    @staticmethod
    def expand_with_ancestors(permissions: list[Permission], ids: set[UUID]) -> set[UUID]:
        by_id = {p.id: p for p in permissions}
        result = set(ids)
        stack = list(ids)
        while stack:
            pid = stack.pop()
            parent = by_id[pid].parent_id if pid in by_id else None
            if parent is not None and parent not in result:
                result.add(parent)
                stack.append(parent)
        return result

    # 查询用户拥有的全部有效权限码（角色绑定去重 + 父链补全 + 过滤停用），供接口鉴权用
    async def get_user_permission_codes(self, user_id: UUID) -> set[str]:
        ids = await self.get_permission_ids_by_user(user_id)
        if not ids:
            return set()
        all_perms = await self.list_all()
        final_ids = self.expand_with_ancestors(all_perms, ids)
        return {p.code for p in all_perms if p.id in final_ids and p.status == 1}

    # 新增权限：写入并提交，刷新后返回权限对象
    async def create(self, permission: Permission) -> Permission:
        self.db.add(permission)
        await self.db.commit()
        stmt = select(Permission).where(Permission.id == permission.id)
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 更新权限：提交变更并刷新，返回更新后的权限对象
    async def update(self, permission: Permission) -> Permission:
        await self.db.commit()
        stmt = select(Permission).where(Permission.id == permission.id)
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 删除权限：从数据库移除并提交事务
    async def delete(self, permission: Permission) -> None:
        await self.db.delete(permission)
        await self.db.commit()
