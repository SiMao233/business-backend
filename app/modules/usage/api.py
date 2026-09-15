"""用量统计 API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。

路由分组：
- 管理端 `/usage/record/*`、`/usage/stat/*`：需功能权限码，数据范围由角色决定（全局 / 仅自己）；
- 个人端 `/usage/my/*`：仅需登录，强制只返回当前用户数据。
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageResult
from app.common.response import ApiResponse, success
from app.core.database import get_db
from app.middleware.authentication import CurrentUserDep, get_current_user
from app.middleware.permission import require_permissions
from app.modules.usage.codes import PermissionCode
from app.modules.usage.schema import (
    UsageOverviewOut,
    UsageRecordOut,
    UsageRecordQuery,
    UsageSummaryOut,
    UsageSummaryQuery,
    UsageTrendOut,
    UsageTrendQuery,
)
from app.modules.usage.service import UsageService

router = APIRouter(prefix="/usage", tags=["用量统计"], dependencies=[Depends(get_current_user)])

DbDep = Annotated[AsyncSession, Depends(get_db)]


def get_service(db: DbDep) -> UsageService:
    """依赖注入工厂：创建绑定数据库会话的 UsageService。"""
    return UsageService(db)


ServiceDep = Annotated[UsageService, Depends(get_service)]

_StartTime = Annotated[
    datetime | None, Query(alias="startTime", description="开始时间（UTC ISO，含）")
]
_EndTime = Annotated[
    datetime | None, Query(alias="endTime", description="结束时间（UTC ISO，不含）")
]


# ---- 管理端：明细 ----
@router.post(
    "/record/list",
    response_model=ApiResponse[PageResult[UsageRecordOut]],
    summary="用量明细列表",
    dependencies=[Depends(require_permissions(PermissionCode.USAGE_RECORD_LIST))],
)
async def list_usage_records(
    query: UsageRecordQuery,
    service: ServiceDep,
    current: CurrentUserDep,
) -> ApiResponse[PageResult[UsageRecordOut]]:
    """分页查询用量明细（数据范围由角色决定）。"""
    return success(data=await service.list_records(query, current))


# ---- 管理端：统计 ----
@router.post(
    "/stat/summary",
    response_model=ApiResponse[UsageSummaryOut],
    summary="用量汇总（按维度分组）",
    dependencies=[Depends(require_permissions(PermissionCode.USAGE_STAT_VIEW))],
)
async def usage_summary(
    query: UsageSummaryQuery,
    service: ServiceDep,
    current: CurrentUserDep,
) -> ApiResponse[UsageSummaryOut]:
    """按 user / agent / model / provider / organization 维度汇总用量。"""
    return success(data=await service.summary(query, current))


@router.post(
    "/stat/trend",
    response_model=ApiResponse[UsageTrendOut],
    summary="用量趋势",
    dependencies=[Depends(require_permissions(PermissionCode.USAGE_STAT_VIEW))],
)
async def usage_trend(
    query: UsageTrendQuery,
    service: ServiceDep,
    current: CurrentUserDep,
) -> ApiResponse[UsageTrendOut]:
    """按业务自然日的用量趋势。"""
    return success(data=await service.trend(query, current))


@router.get(
    "/stat/overview",
    response_model=ApiResponse[UsageOverviewOut],
    summary="用量概览",
    dependencies=[Depends(require_permissions(PermissionCode.USAGE_STAT_VIEW))],
)
async def usage_overview(
    service: ServiceDep,
    current: CurrentUserDep,
    start_time: _StartTime = None,
    end_time: _EndTime = None,
) -> ApiResponse[UsageOverviewOut]:
    """AI 用量 + 业务用量概览（含环比）。"""
    return success(data=await service.overview(start_time, end_time, current))


# ---- 个人端：仅需登录，强制只看自己 ----
@router.get("/my/overview", response_model=ApiResponse[UsageOverviewOut], summary="我的用量概览")
async def my_usage_overview(
    service: ServiceDep,
    current: CurrentUserDep,
    start_time: _StartTime = None,
    end_time: _EndTime = None,
) -> ApiResponse[UsageOverviewOut]:
    """当前登录用户的用量概览。"""
    return success(data=await service.overview(start_time, end_time, current, only_self=True))


@router.post(
    "/my/records",
    response_model=ApiResponse[PageResult[UsageRecordOut]],
    summary="我的用量明细",
)
async def my_usage_records(
    query: UsageRecordQuery,
    service: ServiceDep,
    current: CurrentUserDep,
) -> ApiResponse[PageResult[UsageRecordOut]]:
    """当前登录用户的用量明细（强制只看自己）。"""
    return success(data=await service.list_records(query, current, only_self=True))
