"""模型管理 Repository 层：数据访问（CRUD / 查询）。"""

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.common.pagination import PageParams, PageResult, paginate
from app.common.repository import BaseRepository
from app.modules.model.model import ModelInstance, ModelProvider


class ModelProviderRepository(BaseRepository):
    """模型供应商数据访问仓库。"""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, ModelProvider)

    # 根据主键 ID 查询供应商
    async def get_by_id(self, provider_id: UUID) -> ModelProvider | None:
        stmt = select(ModelProvider).where(ModelProvider.id == provider_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 根据业务编码查询（唯一性校验，可排除自身）
    async def get_by_code(self, code: str, exclude_id: UUID | None = None) -> ModelProvider | None:
        stmt = select(ModelProvider).where(ModelProvider.code == code)
        if exclude_id is not None:
            stmt = stmt.where(ModelProvider.id != exclude_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 分页查询供应商列表，支持关键字 / 状态筛选
    async def list_page(
        self,
        params: PageParams,
        keyword: str | None = None,
        status: int | None = None,
    ) -> PageResult[Any]:
        stmt = select(ModelProvider).order_by(ModelProvider.id.desc())
        if keyword:
            like = f"%{keyword}%"
            stmt = stmt.where(ModelProvider.name.like(like) | ModelProvider.code.like(like))
        if status is not None:
            stmt = stmt.where(ModelProvider.status == status)
        return await paginate(self.db, stmt, params)

    # 新增供应商：写入并提交，刷新后返回对象
    async def create(self, provider: ModelProvider) -> ModelProvider:
        self.db.add(provider)
        await self.db.commit()
        stmt = select(ModelProvider).where(ModelProvider.id == provider.id)
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 更新供应商：提交变更并刷新，返回更新后的对象
    async def update(self, provider: ModelProvider) -> ModelProvider:
        await self.db.commit()
        stmt = select(ModelProvider).where(ModelProvider.id == provider.id)
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 删除供应商：从数据库移除并提交事务
    async def delete(self, provider: ModelProvider) -> None:
        await self.db.delete(provider)
        await self.db.commit()


class ModelInstanceRepository(BaseRepository):
    """模型实例数据访问仓库。"""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, ModelInstance)

    # 根据主键 ID 查询实例（预加载供应商，避免异步懒加载报错）
    async def get_by_id(self, instance_id: UUID) -> ModelInstance | None:
        stmt = (
            select(ModelInstance)
            .options(selectinload(ModelInstance.provider))
            .where(ModelInstance.id == instance_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 分页查询实例列表，支持关键字 / 供应商 / 类型 / 状态筛选
    async def list_page(
        self,
        params: PageParams,
        keyword: str | None = None,
        provider_id: UUID | None = None,
        model_type: str | None = None,
        status: int | None = None,
    ) -> PageResult[Any]:
        stmt = (
            select(ModelInstance)
            .options(selectinload(ModelInstance.provider))
            .order_by(ModelInstance.id.desc())
        )
        if keyword:
            like = f"%{keyword}%"
            stmt = stmt.where(ModelInstance.name.like(like) | ModelInstance.code.like(like))
        if provider_id is not None:
            stmt = stmt.where(ModelInstance.provider_id == provider_id)
        if model_type is not None:
            stmt = stmt.where(ModelInstance.model_type == model_type)
        if status is not None:
            stmt = stmt.where(ModelInstance.status == status)
        return await paginate(self.db, stmt, params)

    # 查询全部启用实例（Agent 创建时的下拉选项）
    async def list_enabled(self) -> list[ModelInstance]:
        stmt = (
            select(ModelInstance)
            .options(selectinload(ModelInstance.provider))
            .where(ModelInstance.status == 1)
            .order_by(ModelInstance.id.desc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # 新增实例：写入并提交，刷新后返回对象
    async def create(self, instance: ModelInstance) -> ModelInstance:
        self.db.add(instance)
        await self.db.commit()
        stmt = (
            select(ModelInstance)
            .options(selectinload(ModelInstance.provider))
            .where(ModelInstance.id == instance.id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 更新实例：提交变更并刷新，返回更新后的对象
    async def update(self, instance: ModelInstance) -> ModelInstance:
        await self.db.commit()
        stmt = (
            select(ModelInstance)
            .options(selectinload(ModelInstance.provider))
            .where(ModelInstance.id == instance.id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 删除实例：从数据库移除并提交事务
    async def delete(self, instance: ModelInstance) -> None:
        await self.db.delete(instance)
        await self.db.commit()
