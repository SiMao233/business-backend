"""组织管理 Repository 层：数据访问（CRUD / 查询）。"""

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.common.pagination import PageParams, PageResult, paginate
from app.common.repository import BaseRepository
from app.modules.organization.model import Organization


class OrganizationRepository(BaseRepository):
    """组织数据访问仓库。"""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, Organization)

    # 根据主键 ID 查询组织
    async def get_by_id(self, org_id: UUID) -> Organization | None:
        stmt = select(Organization).where(Organization.id == org_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 根据业务编码查询（唯一性校验，可排除自身）
    async def get_by_code(self, code: str, exclude_id: UUID | None = None) -> Organization | None:
        stmt = select(Organization).where(Organization.code == code)
        if exclude_id is not None:
            stmt = stmt.where(Organization.id != exclude_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 全量组织列表（树组装用），显式 selectinload children 避免异步懒加载
    async def list_all(self) -> list[Organization]:
        stmt = (
            select(Organization)
            .options(selectinload(Organization.children))
            .order_by(Organization.name.asc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # 分页查询组织列表，支持关键字 / 状态筛选
    async def list_page(
        self,
        params: PageParams,
        keyword: str | None = None,
        status: int | None = None,
    ) -> PageResult[Any]:
        stmt = select(Organization).order_by(Organization.id.desc())
        if keyword:
            like = f"%{keyword}%"
            stmt = stmt.where(Organization.name.like(like) | Organization.code.like(like))
        if status is not None:
            stmt = stmt.where(Organization.status == status)
        return await paginate(self.db, stmt, params)

    # 新增组织：写入并提交，刷新后返回对象
    async def create(self, org: Organization) -> Organization:
        self.db.add(org)
        await self.db.commit()
        stmt = select(Organization).where(Organization.id == org.id)
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 更新组织：提交变更并刷新，返回更新后的对象
    async def update(self, org: Organization) -> Organization:
        await self.db.commit()
        stmt = select(Organization).where(Organization.id == org.id)
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 删除组织：从数据库移除并提交事务
    async def delete(self, org: Organization) -> None:
        await self.db.delete(org)
        await self.db.commit()
