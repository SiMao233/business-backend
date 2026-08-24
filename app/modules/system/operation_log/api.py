"""系统操作日志 API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。"""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageResult
from app.common.response import ApiResponse, success
from app.core.database import get_db
from app.middleware.authentication import get_current_user
from app.middleware.permission import require_permissions
from app.modules.system.operation_log.schema import OperationLogOut, OperationLogQuery
from app.modules.system.operation_log.service import OperationLogService
from app.modules.system.permission.codes import PermissionCode

router = APIRouter(
    prefix="/operation-log",
    tags=["系统-操作日志"],
    dependencies=[Depends(get_current_user)],
)

DbDep = Annotated[AsyncSession, Depends(get_db)]


# 依赖注入工厂：创建 OperationLogService 实例（绑定数据库会话）
def get_service(db: DbDep) -> OperationLogService:
    return OperationLogService(db)


# 操作日志列表接口：分页查询，支持按模块/动作/操作人/状态筛选
@router.post(
    "/list",
    response_model=ApiResponse[PageResult[OperationLogOut]],
    summary="操作日志列表",
    dependencies=[Depends(require_permissions(PermissionCode.OPERATION_LOG_LIST))],
)
async def list_operation_logs(
    query: OperationLogQuery,
    service: Annotated[OperationLogService, Depends(get_service)],
) -> ApiResponse[PageResult[OperationLogOut]]:
    return success(data=await service.list_page(query))