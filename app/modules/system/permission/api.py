"""系统权限管理 API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageResult
from app.common.response import ApiResponse, success
from app.core.database import get_db
from app.middleware.authentication import CurrentUserDep, get_current_user
from app.middleware.permission import require_permissions
from app.modules.system.operation_log.service import OperationLogService
from app.modules.system.permission.codes import PermissionCode
from app.modules.system.permission.schema import (
    PermissionCreate,
    PermissionNode,
    PermissionOut,
    PermissionQuery,
    PermissionUpdate,
    UserPermissionOut,
)
from app.modules.system.permission.service import PermissionService

router = APIRouter(prefix="/permission", tags=["系统-权限管理"], dependencies=[Depends(get_current_user)])

DbDep = Annotated[AsyncSession, Depends(get_db)]


# 依赖注入工厂：创建 PermissionService 实例（绑定数据库会话）
def get_service(db: DbDep) -> PermissionService:
    return PermissionService(db)


# 权限列表接口：分页查询，支持关键字/类型/状态筛选
@router.post(
    "/list",
    response_model=ApiResponse[PageResult[PermissionOut]],
    summary="权限列表",
    dependencies=[Depends(require_permissions(PermissionCode.PERMISSION_LIST))],
)
async def list_permissions(
    query: PermissionQuery,
    service: Annotated[PermissionService, Depends(get_service)],
) -> ApiResponse[PageResult[PermissionOut]]:
    return success(data=await service.list_page(query))


# 权限树接口：全量树形返回（管理端展示 / 分配用）
@router.get(
    "/tree",
    response_model=ApiResponse[list[PermissionNode]],
    summary="权限树",
    dependencies=[Depends(require_permissions(PermissionCode.PERMISSION_LIST))],
)
async def permission_tree(
    service: Annotated[PermissionService, Depends(get_service)],
) -> ApiResponse[list[PermissionNode]]:
    return success(data=await service.tree())


# 当前用户聚合权限接口：权限码集合 + 菜单树（供前端渲染菜单/按钮鉴权）
# 注意：所有登录用户均可访问，不限制具体权限码
@router.get("/mine", response_model=ApiResponse[UserPermissionOut], summary="我的权限")
async def my_permissions(
    current: CurrentUserDep,
    service: Annotated[PermissionService, Depends(get_service)],
) -> ApiResponse[UserPermissionOut]:
    assert current.user_id is not None  # 鉴权通过后必有用户 ID
    return success(data=await service.mine(current.user_id))


# 权限详情接口：根据 ID 查询单个权限
@router.get(
    "/{permission_id}",
    response_model=ApiResponse[PermissionOut],
    summary="权限详情",
    dependencies=[Depends(require_permissions(PermissionCode.PERMISSION_LIST))],
)
async def get_permission(
    permission_id: UUID,
    service: Annotated[PermissionService, Depends(get_service)],
) -> ApiResponse[PermissionOut]:
    return success(data=await service.get(permission_id))


# 创建权限接口：接收创建参数并新增权限
@router.post(
    "/create",
    response_model=ApiResponse[PermissionOut],
    summary="创建权限",
    dependencies=[Depends(require_permissions(PermissionCode.PERMISSION_CREATE))],
)
async def create_permission(
    req: PermissionCreate,
    service: Annotated[PermissionService, Depends(get_service)],
) -> ApiResponse[PermissionOut]:
    return success(data=await service.create(req))


# 更新权限接口：根据 ID 更新权限信息
@router.put(
    "/{permission_id}",
    response_model=ApiResponse[PermissionOut],
    summary="更新权限",
    dependencies=[Depends(require_permissions(PermissionCode.PERMISSION_UPDATE))],
)
async def update_permission(
    permission_id: UUID,
    req: PermissionUpdate,
    service: Annotated[PermissionService, Depends(get_service)],
) -> ApiResponse[PermissionOut]:
    return success(data=await service.update(permission_id, req), message="更新成功")


# 删除权限接口：根据 ID 删除权限
@router.delete(
    "/{permission_id}",
    response_model=ApiResponse[None],
    summary="删除权限",
    dependencies=[Depends(require_permissions(PermissionCode.PERMISSION_DELETE))],
)
async def delete_permission(
    permission_id: UUID,
    request: Request,
    current: CurrentUserDep,
    db: DbDep,
    service: Annotated[PermissionService, Depends(get_service)],
) -> ApiResponse[None]:
    deleted = await service.delete(permission_id, current)
    await OperationLogService(db).record(
        user=current, module="permission", action="delete", target_id=permission_id,
        detail={"code": deleted.code, "name": deleted.name}, request=request,
    )
    return success(message="删除成功")
