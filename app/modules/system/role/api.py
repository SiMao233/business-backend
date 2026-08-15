"""系统角色管理 API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。

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
from app.modules.system.permission.codes import PermissionCode
from app.modules.system.role.schema import (
    RoleCreate,
    RoleOut,
    RolePermissionsReq,
    RoleQuery,
    RoleUpdate,
)
from app.modules.system.role.service import RoleService

router = APIRouter(prefix="/role", tags=["系统-角色管理"], dependencies=[Depends(get_current_user)])

DbDep = Annotated[AsyncSession, Depends(get_db)]

# 依赖注入工厂：创建 RoleService 实例（绑定数据库会话）
def get_service(db: DbDep) -> RoleService:
    return RoleService(db)

# 角色列表接口：分页查询，支持按关键字模糊搜索
@router.post(
    "/list",
    response_model=ApiResponse[PageResult[RoleOut]],
    summary="角色列表",
    dependencies=[Depends(require_permissions(PermissionCode.ROLE_LIST))],
)
async def list_roles(
    query: RoleQuery,
    service: Annotated[RoleService, Depends(get_service)],
) -> ApiResponse[PageResult[RoleOut]]:
    return success(data=await service.list_page(query))


# 角色详情接口：根据 ID 查询单个角色
@router.get(
    "/{role_id}",
    response_model=ApiResponse[RoleOut],
    summary="角色详情",
    dependencies=[Depends(require_permissions(PermissionCode.ROLE_LIST))],
)
async def get_role(
    role_id: UUID, service: Annotated[RoleService, Depends(get_service)]
) -> ApiResponse[RoleOut]:
    return success(data=await service.get(role_id))


# 创建角色接口：接收创建参数并新增角色
@router.post(
    "/create",
    response_model=ApiResponse[RoleOut],
    summary="创建角色",
    dependencies=[Depends(require_permissions(PermissionCode.ROLE_CREATE))],
)
async def create_role(
    req: RoleCreate, service: Annotated[RoleService, Depends(get_service)]
) -> ApiResponse[RoleOut]:
    return success(data=await service.create(req))


# 分配权限接口：全量覆盖式绑定角色权限（提交子节点时自动补全父节点）
@router.put(
    "/{role_id}/permissions",
    response_model=ApiResponse[RoleOut],
    summary="分配权限",
    dependencies=[Depends(require_permissions(PermissionCode.ROLE_ASSIGN_PERMISSION))],
)
async def assign_permissions(
    role_id: UUID,
    req: RolePermissionsReq,
    service: Annotated[RoleService, Depends(get_service)],
) -> ApiResponse[RoleOut]:
    return success(data=await service.assign_permissions(role_id, req), message="分配成功")


# 更新角色接口：根据 ID 更新角色信息
@router.put(
    "/{role_id}",
    response_model=ApiResponse[RoleOut],
    summary="更新角色",
    dependencies=[Depends(require_permissions(PermissionCode.ROLE_UPDATE))],
)
async def update_role(
    role_id: UUID, req: RoleUpdate, service: Annotated[RoleService, Depends(get_service)]
) -> ApiResponse[RoleOut]:
    return success(data=await service.update(role_id, req),message="更新成功")


# 删除角色接口：根据 ID 删除角色
@router.delete(
    "/{role_id}",
    response_model=ApiResponse[None],
    summary="删除角色",
    dependencies=[Depends(require_permissions(PermissionCode.ROLE_DELETE))],
)
async def delete_role(
    role_id: UUID, service: Annotated[RoleService, Depends(get_service)]
) -> ApiResponse[None]:
    await service.delete(role_id)
    return success(message="删除成功")
