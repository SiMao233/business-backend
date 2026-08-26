"""模型管理 Service 层：业务规则、权限判断、流程编排。"""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageParams, PageResult
from app.core.exceptions import BizError, NotFoundError
from app.modules.model.model import ModelInstance, ModelProvider
from app.modules.model.repository import ModelInstanceRepository, ModelProviderRepository
from app.modules.model.schema import (
    ModelInstanceCreate,
    ModelInstanceOption,
    ModelInstanceOut,
    ModelInstanceQuery,
    ModelInstanceUpdate,
    ModelProviderCreate,
    ModelProviderOut,
    ModelProviderQuery,
    ModelProviderUpdate,
)


class ModelService:
    """模型管理服务（供应商 + 实例）。"""

    def __init__(self, db: AsyncSession) -> None:
        self.provider_repo = ModelProviderRepository(db)
        self.instance_repo = ModelInstanceRepository(db)

    # ---- 供应商 ----
    async def list_providers(self, query: ModelProviderQuery) -> PageResult[ModelProviderOut]:
        page_params = PageParams(page=query.page, pageSize=query.pageSize)
        page = await self.provider_repo.list_page(
            page_params, keyword=query.keyword, status=query.status
        )
        return PageResult(
            list=[self._provider_out(item) for item in page.list],
            total=page.total,
            page=page.page,
            pageSize=page.pageSize,
        )

    async def get_provider(self, provider_id: UUID) -> ModelProviderOut:
        provider = await self.provider_repo.get_by_id(provider_id)
        if not provider:
            raise NotFoundError("模型供应商不存在")
        return self._provider_out(provider)

    async def create_provider(self, req: ModelProviderCreate) -> ModelProviderOut:
        if await self.provider_repo.get_by_code(req.code):
            raise BizError("供应商编码已存在")
        provider = ModelProvider(
            name=req.name,
            code=req.code,
            base_url=req.base_url,
            api_key=req.api_key,
            status=req.status,
        )
        return self._provider_out(await self.provider_repo.create(provider))

    async def update_provider(self, provider_id: UUID, req: ModelProviderUpdate) -> ModelProviderOut:
        provider = await self.provider_repo.get_by_id(provider_id)
        if not provider:
            raise NotFoundError("模型供应商不存在")
        if provider.is_builtin and req.status == 0:
            raise BizError("内置供应商不允许禁用")
        if req.name is not None:
            provider.name = req.name
        if req.base_url is not None:
            provider.base_url = req.base_url
        if req.api_key is not None:
            provider.api_key = req.api_key
        if req.status is not None:
            provider.status = req.status
        return self._provider_out(await self.provider_repo.update(provider))

    async def delete_provider(self, provider_id: UUID) -> None:
        provider = await self.provider_repo.get_by_id(provider_id)
        if not provider:
            raise NotFoundError("模型供应商不存在")
        if provider.is_builtin:
            raise BizError("内置供应商不允许删除")
        await self.provider_repo.delete(provider)

    # ---- 实例 ----
    async def list_instances(self, query: ModelInstanceQuery) -> PageResult[ModelInstanceOut]:
        page_params = PageParams(page=query.page, pageSize=query.pageSize)
        page = await self.instance_repo.list_page(
            page_params,
            keyword=query.keyword,
            provider_id=query.provider_id,
            model_type=query.model_type.value if query.model_type else None,
            status=query.status,
        )
        return PageResult(
            list=[self._instance_out(item) for item in page.list],
            total=page.total,
            page=page.page,
            pageSize=page.pageSize,
        )

    async def get_instance(self, instance_id: UUID) -> ModelInstanceOut:
        instance = await self.instance_repo.get_by_id(instance_id)
        if not instance:
            raise NotFoundError("模型实例不存在")
        return self._instance_out(instance)

    async def create_instance(self, req: ModelInstanceCreate) -> ModelInstanceOut:
        provider = await self.provider_repo.get_by_id(req.provider_id)
        if not provider:
            raise NotFoundError("模型供应商不存在")
        instance = ModelInstance(
            provider_id=req.provider_id,
            name=req.name,
            code=req.code,
            model_type=req.model_type.value,
            max_tokens=req.max_tokens,
            status=req.status,
        )
        return self._instance_out(await self.instance_repo.create(instance))

    async def update_instance(self, instance_id: UUID, req: ModelInstanceUpdate) -> ModelInstanceOut:
        instance = await self.instance_repo.get_by_id(instance_id)
        if not instance:
            raise NotFoundError("模型实例不存在")
        if req.name is not None:
            instance.name = req.name
        if req.code is not None:
            instance.code = req.code
        if req.model_type is not None:
            instance.model_type = req.model_type.value
        if req.max_tokens is not None:
            instance.max_tokens = req.max_tokens
        if req.status is not None:
            instance.status = req.status
        return self._instance_out(await self.instance_repo.update(instance))

    async def delete_instance(self, instance_id: UUID) -> None:
        instance = await self.instance_repo.get_by_id(instance_id)
        if not instance:
            raise NotFoundError("模型实例不存在")
        await self.instance_repo.delete(instance)

    # 全部启用实例的下拉选项（Agent 创建时选择模型用）
    async def list_instance_options(self) -> list[ModelInstanceOption]:
        instances = await self.instance_repo.list_enabled()
        return [self._instance_option(item) for item in instances]

    # ---- 出参转换 ----
    @staticmethod
    def _provider_out(provider: ModelProvider) -> ModelProviderOut:
        out = ModelProviderOut.model_validate(provider)
        out.has_api_key = bool(provider.api_key)
        return out

    @staticmethod
    def _instance_out(instance: ModelInstance) -> ModelInstanceOut:
        out = ModelInstanceOut.model_validate(instance)
        out.provider_name = instance.provider.name if instance.provider else None
        return out

    @staticmethod
    def _instance_option(instance: ModelInstance) -> ModelInstanceOption:
        out = ModelInstanceOption.model_validate(instance)
        out.provider_name = instance.provider.name if instance.provider else None
        return out
