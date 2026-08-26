"""模型管理 API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。

本层禁止直接操作数据库或编写复杂业务逻辑。
路由风格：动作前置、id 放最后（如 /model/provider/list）。
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageResult
from app.common.response import ApiResponse, success
from app.core.database import get_db
from app.middleware.authentication import get_current_user
from app.middleware.permission import require_permissions
from app.modules.model.codes import PermissionCode
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
from app.modules.model.service import ModelService

router = APIRouter(prefix="/model", tags=["模型管理"], dependencies=[Depends(get_current_user)])

DbDep = Annotated[AsyncSession, Depends(get_db)]


# 依赖注入工厂：创建 ModelService 实例（绑定数据库会话）
def get_service(db: DbDep) -> ModelService:
    return ModelService(db)


# ---- 模型供应商 ----
@router.post(
    "/provider/list",
    response_model=ApiResponse[PageResult[ModelProviderOut]],
    summary="模型供应商列表",
    dependencies=[Depends(require_permissions(PermissionCode.MODEL_PROVIDER_LIST))],
)
async def list_providers(
    query: ModelProviderQuery,
    service: Annotated[ModelService, Depends(get_service)],
) -> ApiResponse[PageResult[ModelProviderOut]]:
    return success(data=await service.list_providers(query))


@router.post(
    "/provider/create",
    response_model=ApiResponse[ModelProviderOut],
    summary="创建模型供应商",
    dependencies=[Depends(require_permissions(PermissionCode.MODEL_PROVIDER_CREATE))],
)
async def create_provider(
    req: ModelProviderCreate,
    service: Annotated[ModelService, Depends(get_service)],
) -> ApiResponse[ModelProviderOut]:
    return success(data=await service.create_provider(req), message="创建成功")


@router.get(
    "/provider/{provider_id}",
    response_model=ApiResponse[ModelProviderOut],
    summary="模型供应商详情",
    dependencies=[Depends(require_permissions(PermissionCode.MODEL_PROVIDER_LIST))],
)
async def get_provider(
    provider_id: UUID,
    service: Annotated[ModelService, Depends(get_service)],
) -> ApiResponse[ModelProviderOut]:
    return success(data=await service.get_provider(provider_id))


@router.put(
    "/provider/{provider_id}",
    response_model=ApiResponse[ModelProviderOut],
    summary="更新模型供应商",
    dependencies=[Depends(require_permissions(PermissionCode.MODEL_PROVIDER_UPDATE))],
)
async def update_provider(
    provider_id: UUID,
    req: ModelProviderUpdate,
    service: Annotated[ModelService, Depends(get_service)],
) -> ApiResponse[ModelProviderOut]:
    return success(data=await service.update_provider(provider_id, req), message="更新成功")


@router.delete(
    "/provider/{provider_id}",
    response_model=ApiResponse[None],
    summary="删除模型供应商",
    dependencies=[Depends(require_permissions(PermissionCode.MODEL_PROVIDER_DELETE))],
)
async def delete_provider(
    provider_id: UUID,
    service: Annotated[ModelService, Depends(get_service)],
) -> ApiResponse[None]:
    await service.delete_provider(provider_id)
    return success(message="删除成功")


# ---- 模型实例 ----
# 注意：静态路径 /instance/options 必须注册在 /instance/{instance_id} 之前，避免被 UUID 参数吞掉
@router.get(
    "/instance/options",
    response_model=ApiResponse[list[ModelInstanceOption]],
    summary="模型实例下拉选项",
    dependencies=[Depends(require_permissions(PermissionCode.MODEL_INSTANCE_LIST))],
)
async def list_instance_options(
    service: Annotated[ModelService, Depends(get_service)],
) -> ApiResponse[list[ModelInstanceOption]]:
    return success(data=await service.list_instance_options())


@router.post(
    "/instance/list",
    response_model=ApiResponse[PageResult[ModelInstanceOut]],
    summary="模型实例列表",
    dependencies=[Depends(require_permissions(PermissionCode.MODEL_INSTANCE_LIST))],
)
async def list_instances(
    query: ModelInstanceQuery,
    service: Annotated[ModelService, Depends(get_service)],
) -> ApiResponse[PageResult[ModelInstanceOut]]:
    return success(data=await service.list_instances(query))


@router.post(
    "/instance/create",
    response_model=ApiResponse[ModelInstanceOut],
    summary="创建模型实例",
    dependencies=[Depends(require_permissions(PermissionCode.MODEL_INSTANCE_CREATE))],
)
async def create_instance(
    req: ModelInstanceCreate,
    service: Annotated[ModelService, Depends(get_service)],
) -> ApiResponse[ModelInstanceOut]:
    return success(data=await service.create_instance(req), message="创建成功")


@router.get(
    "/instance/{instance_id}",
    response_model=ApiResponse[ModelInstanceOut],
    summary="模型实例详情",
    dependencies=[Depends(require_permissions(PermissionCode.MODEL_INSTANCE_LIST))],
)
async def get_instance(
    instance_id: UUID,
    service: Annotated[ModelService, Depends(get_service)],
) -> ApiResponse[ModelInstanceOut]:
    return success(data=await service.get_instance(instance_id))


@router.put(
    "/instance/{instance_id}",
    response_model=ApiResponse[ModelInstanceOut],
    summary="更新模型实例",
    dependencies=[Depends(require_permissions(PermissionCode.MODEL_INSTANCE_UPDATE))],
)
async def update_instance(
    instance_id: UUID,
    req: ModelInstanceUpdate,
    service: Annotated[ModelService, Depends(get_service)],
) -> ApiResponse[ModelInstanceOut]:
    return success(data=await service.update_instance(instance_id, req), message="更新成功")


@router.delete(
    "/instance/{instance_id}",
    response_model=ApiResponse[None],
    summary="删除模型实例",
    dependencies=[Depends(require_permissions(PermissionCode.MODEL_INSTANCE_DELETE))],
)
async def delete_instance(
    instance_id: UUID,
    service: Annotated[ModelService, Depends(get_service)],
) -> ApiResponse[None]:
    await service.delete_instance(instance_id)
    return success(message="删除成功")
