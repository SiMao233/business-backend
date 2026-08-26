"""组织管理 API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。

本层禁止直接操作数据库或编写复杂业务逻辑。
路由风格：动作前置、id 放最后（如 /organization/list）。
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
from app.modules.organization.codes import PermissionCode
from app.modules.organization.schema import (
    OrganizationCreate,
    OrganizationNode,
    OrganizationOut,
    OrganizationQuery,
    OrganizationUpdate,
)
from app.modules.organization.service import OrganizationService

router = APIRouter(prefix="/organization", tags=["组织管理"], dependencies=[Depends(get_current_user)])

DbDep = Annotated[AsyncSession, Depends(get_db)]


# 依赖注入工厂：创建 OrganizationService 实例（绑定数据库会话）
def get_service(db: DbDep) -> OrganizationService:
    return OrganizationService(db)


# 组织树接口：静态路径 /tree 必须注册在 /{organization_id} 之前，避免被 UUID 参数吞掉
@router.get(
    "/tree",
    response_model=ApiResponse[list[OrganizationNode]],
    summary="组织树",
    dependencies=[Depends(require_permissions(PermissionCode.ORGANIZATION_LIST))],
)
async def organization_tree(
    service: Annotated[OrganizationService, Depends(get_service)],
) -> ApiResponse[list[OrganizationNode]]:
    return success(data=await service.tree())


# 组织列表接口：分页查询
@router.post(
    "/list",
    response_model=ApiResponse[PageResult[OrganizationOut]],
    summary="组织列表",
    dependencies=[Depends(require_permissions(PermissionCode.ORGANIZATION_LIST))],
)
async def list_organizations(
    query: OrganizationQuery,
    service: Annotated[OrganizationService, Depends(get_service)],
) -> ApiResponse[PageResult[OrganizationOut]]:
    return success(data=await service.list_page(query))


# 创建组织接口
@router.post(
    "/create",
    response_model=ApiResponse[OrganizationOut],
    summary="创建组织",
    dependencies=[Depends(require_permissions(PermissionCode.ORGANIZATION_CREATE))],
)
async def create_organization(
    req: OrganizationCreate,
    service: Annotated[OrganizationService, Depends(get_service)],
) -> ApiResponse[OrganizationOut]:
    return success(data=await service.create(req), message="创建成功")


# 组织详情接口
@router.get(
    "/{organization_id}",
    response_model=ApiResponse[OrganizationOut],
    summary="组织详情",
    dependencies=[Depends(require_permissions(PermissionCode.ORGANIZATION_LIST))],
)
async def get_organization(
    organization_id: UUID,
    service: Annotated[OrganizationService, Depends(get_service)],
) -> ApiResponse[OrganizationOut]:
    return success(data=await service.get(organization_id))


# 更新组织接口
@router.put(
    "/{organization_id}",
    response_model=ApiResponse[OrganizationOut],
    summary="更新组织",
    dependencies=[Depends(require_permissions(PermissionCode.ORGANIZATION_UPDATE))],
)
async def update_organization(
    organization_id: UUID,
    req: OrganizationUpdate,
    service: Annotated[OrganizationService, Depends(get_service)],
) -> ApiResponse[OrganizationOut]:
    return success(data=await service.update(organization_id, req), message="更新成功")


# 删除组织接口
@router.delete(
    "/{organization_id}",
    response_model=ApiResponse[None],
    summary="删除组织",
    dependencies=[Depends(require_permissions(PermissionCode.ORGANIZATION_DELETE))],
)
async def delete_organization(
    organization_id: UUID,
    service: Annotated[OrganizationService, Depends(get_service)],
) -> ApiResponse[None]:
    await service.delete(organization_id)
    return success(message="删除成功")
