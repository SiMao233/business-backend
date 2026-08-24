"""系统用户管理 API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageResult
from app.common.response import ApiResponse, success
from app.core.database import get_db
from app.core.redis import get_redis
from app.middleware.authentication import CurrentUserDep, get_current_user
from app.middleware.permission import require_permissions
from app.modules.system.operation_log.service import OperationLogService
from app.modules.system.permission.codes import PermissionCode
from app.modules.system.user.schema import (
    UserCreate,
    UserOut,
    UserQuery,
    UserResetPassword,
    UserUpdate,
)
from app.modules.system.user.service import UserService

router = APIRouter(prefix="/user", tags=["系统-用户管理"], dependencies=[Depends(get_current_user)])

DbDep = Annotated[AsyncSession, Depends(get_db)]
RedisDep = Annotated[Redis, Depends(get_redis)]


# 依赖注入工厂：创建 UserService 实例（绑定数据库会话与 Redis）
def get_service(db: DbDep, redis: RedisDep) -> UserService:
    return UserService(db, redis)


# 用户列表接口：分页查询，支持关键字/状态/角色筛选
@router.post(
    "/list",
    response_model=ApiResponse[PageResult[UserOut]],
    summary="用户列表",
    dependencies=[Depends(require_permissions(PermissionCode.USER_LIST))],
)
async def list_users(
    query: UserQuery,
    service: Annotated[UserService, Depends(get_service)],
) -> ApiResponse[PageResult[UserOut]]:
    return success(data=await service.list_page(query))


# 用户详情接口：根据 ID 查询单个用户
@router.get(
    "/{user_id}",
    response_model=ApiResponse[UserOut],
    summary="用户详情",
    dependencies=[Depends(require_permissions(PermissionCode.USER_LIST))],
)
async def get_user(
    user_id: UUID, service: Annotated[UserService, Depends(get_service)]
) -> ApiResponse[UserOut]:
    return success(data=await service.get(user_id))


# 创建用户接口：接收创建参数并新增用户
@router.post(
    "/create",
    response_model=ApiResponse[UserOut],
    summary="创建用户",
    dependencies=[Depends(require_permissions(PermissionCode.USER_CREATE))],
)
async def create_user(
    req: UserCreate, service: Annotated[UserService, Depends(get_service)]
) -> ApiResponse[UserOut]:
    return success(data=await service.create(req))


# 更新用户接口：根据 ID 更新用户信息
@router.put(
    "/{user_id}",
    response_model=ApiResponse[UserOut],
    summary="更新用户",
    dependencies=[Depends(require_permissions(PermissionCode.USER_UPDATE))],
)
async def update_user(
    user_id: UUID, req: UserUpdate, service: Annotated[UserService, Depends(get_service)]
) -> ApiResponse[UserOut]:
    return success(data=await service.update(user_id, req), message="更新成功")


# 重置密码接口：管理员重置指定用户密码（user_id 放在 body 中）
@router.post(
    "/resetPassword",
    response_model=ApiResponse[None],
    summary="重置密码",
    dependencies=[Depends(require_permissions(PermissionCode.USER_RESET_PASSWORD))],
)
async def reset_password(
    req: UserResetPassword,
    service: Annotated[UserService, Depends(get_service)],
) -> ApiResponse[None]:
    await service.reset_password(req)
    return success(message="密码重置成功")


# 删除用户接口：根据 ID 删除用户
@router.delete(
    "/{user_id}",
    response_model=ApiResponse[None],
    summary="删除用户",
    dependencies=[Depends(require_permissions(PermissionCode.USER_DELETE))],
)
async def delete_user(
    user_id: UUID,
    request: Request,
    current: CurrentUserDep,
    db: DbDep,
    service: Annotated[UserService, Depends(get_service)],
) -> ApiResponse[None]:
    deleted = await service.delete(user_id, current)
    await OperationLogService(db).record(
        user=current, module="user", action="delete", target_id=user_id,
        detail={"username": deleted.username}, request=request,
    )
    return success(message="删除成功")
