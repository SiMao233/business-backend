"""用量统计 Service 层：用量写入、数据范围解析、明细 / 汇总 / 趋势 / 概览查询。"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import TrendGranularity, UsageCallType
from app.common.pagination import PageParams, PageResult
from app.core.exceptions import ValidateError
from app.middleware.authentication import UserContext
from app.modules.usage.codes import PermissionCode
from app.modules.usage.model import AiUsageRecord
from app.modules.usage.repository import UsageRepository
from app.modules.usage.schema import (
    UsageAiOverviewOut,
    UsageBizOverviewOut,
    UsageCompareOut,
    UsageOverviewOut,
    UsageRecordOut,
    UsageRecordQuery,
    UsageSummaryItemOut,
    UsageSummaryOut,
    UsageSummaryQuery,
    UsageTotalOut,
    UsageTrendItemOut,
    UsageTrendOut,
    UsageTrendQuery,
)
from app.modules.usage.utils import business_date, calc_cost, normalize_range


@dataclass
class UsageSource:
    """用量记录来源：由调用方（AI 对话）填充的维度快照与用量。

    维度字段一律传「快照值」，使统计查询无需 JOIN，且源数据删除后仍可追溯。
    """

    # 用量
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # 维度快照
    user_id: UUID | None = None
    username: str | None = None
    agent_id: UUID | None = None
    agent_name: str | None = None
    organization_id: UUID | None = None
    model_instance_id: UUID | None = None
    provider_id: UUID | None = None
    provider_code: str | None = None
    model_code: str | None = None
    conversation_id: UUID | None = None
    message_id: UUID | None = None

    # 单价快照（元 / 千 token）与上下文
    input_price: Decimal | None = None
    output_price: Decimal | None = None
    call_type: str = UsageCallType.CHAT.value
    latency_ms: int | None = None
    status: int = 1
    request_id: str | None = None


def _pct(current: int, previous: int) -> float | None:
    """环比百分比；上期为 0 时返回 None（无法计算）。"""
    if previous <= 0:
        return None
    return round((current - previous) / previous * 100, 2)


class UsageService:
    """用量统计服务。"""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repo = UsageRepository(db)

    # ---- 写入（供 AI 对话调用）----
    async def record(self, source: UsageSource) -> None:
        """追加写入一条用量明细。

        独立提交、异常仅记日志：用量统计属于旁路能力，**绝不能影响对话主流程**。
        """
        try:
            record = AiUsageRecord(
                user_id=source.user_id,
                username=source.username,
                agent_id=source.agent_id,
                agent_name=source.agent_name,
                organization_id=source.organization_id,
                model_instance_id=source.model_instance_id,
                provider_id=source.provider_id,
                provider_code=source.provider_code,
                model_code=source.model_code,
                conversation_id=source.conversation_id,
                message_id=source.message_id,
                input_tokens=source.input_tokens,
                output_tokens=source.output_tokens,
                total_tokens=source.total_tokens,
                input_price=source.input_price,
                output_price=source.output_price,
                total_cost=calc_cost(
                    source.input_tokens,
                    source.output_tokens,
                    source.input_price,
                    source.output_price,
                ),
                call_type=source.call_type,
                latency_ms=source.latency_ms,
                status=source.status,
                request_id=source.request_id,
                usage_date=business_date(),
            )
            await self.repo.create(record)
        except Exception:  # noqa: BLE001 - 旁路能力，写入失败不得影响主流程
            logger.exception(
                "用量明细落库失败 user_id={} agent_id={} model={}",
                source.user_id,
                source.agent_id,
                source.model_code,
            )

    # ---- 数据范围 ----
    async def _scope_user_id(self, current: UserContext) -> UUID | None:
        """解析数据范围：返回 None 表示可看全局；否则仅能查看该用户数据。

        超级管理员或持有 `usage:stat:all` 权限码者可看全局。
        """
        user_id = current.user_id
        if user_id is None:
            raise ValidateError("无法识别当前用户")
        # 延迟导入：避免与 system.permission 模块产生导入环
        from app.modules.system.permission.repository import PermissionRepository

        repo = PermissionRepository(self.db)
        if await repo.is_superuser(user_id):
            return None
        codes = await repo.get_user_permission_codes(user_id)
        if PermissionCode.USAGE_STAT_ALL in codes:
            return None
        return user_id

    # ---- 明细 ----
    async def list_records(
        self, query: UsageRecordQuery, current: UserContext, *, only_self: bool = False
    ) -> PageResult[UsageRecordOut]:
        start, end = normalize_range(query.start_time, query.end_time)
        scope = await self._resolve_scope(current, only_self)
        page = await self.repo.list_page(
            PageParams(page=query.page, pageSize=query.pageSize),
            start=start,
            end=end,
            scope_user_id=scope,
            user_id=query.user_id,
            agent_id=query.agent_id,
            model_instance_id=query.model_instance_id,
            organization_id=query.organization_id,
            call_type=query.call_type.value if query.call_type else None,
            status=query.status,
        )
        return PageResult(
            list=[UsageRecordOut.model_validate(item) for item in page.list],
            total=page.total,
            page=page.page,
            pageSize=page.pageSize,
        )

    # ---- 汇总 ----
    async def summary(
        self, query: UsageSummaryQuery, current: UserContext, *, only_self: bool = False
    ) -> UsageSummaryOut:
        start, end = normalize_range(query.start_time, query.end_time)
        scope = await self._resolve_scope(current, only_self)
        dimension = query.dimension.value
        rows = await self.repo.summary_by(
            dimension, start=start, end=end, scope_user_id=scope, top=query.top
        )
        total = await self.repo.summary_total(start=start, end=end, scope_user_id=scope)
        return UsageSummaryOut(
            dimension=dimension,
            list=[UsageSummaryItemOut.model_validate(row) for row in rows],
            total=UsageTotalOut.model_validate(total),
        )

    # ---- 趋势 ----
    async def trend(
        self, query: UsageTrendQuery, current: UserContext, *, only_self: bool = False
    ) -> UsageTrendOut:
        if query.granularity != TrendGranularity.DAY:
            # 小时粒度需按业务时区切桶，第一期先不做（避免依赖 MySQL 时区表）
            raise ValidateError("当前仅支持 day 粒度趋势，hour 粒度将在后续版本提供")
        start, end = normalize_range(query.start_time, query.end_time)
        scope = await self._resolve_scope(current, only_self)
        rows = await self.repo.trend_by_day(start=start, end=end, scope_user_id=scope)
        items: list[UsageTrendItemOut] = []
        for row in rows:
            data = dict(row)
            data["date"] = str(data.pop("stat_date"))
            items.append(UsageTrendItemOut.model_validate(data))
        return UsageTrendOut(granularity=query.granularity.value, list=items)

    # ---- 概览 ----
    async def overview(
        self,
        start: datetime | None,
        end: datetime | None,
        current: UserContext,
        *,
        only_self: bool = False,
    ) -> UsageOverviewOut:
        start_utc, end_utc = normalize_range(start, end)
        scope = await self._resolve_scope(current, only_self)

        ai_row = await self.repo.overview_ai(start=start_utc, end=end_utc, scope_user_id=scope)
        biz_row = await self.repo.biz_counts(start=start_utc, end=end_utc, scope_user_id=scope)

        # 环比：与「上一等长周期」比较
        span = end_utc - start_utc
        prev_row = await self.repo.overview_ai(
            start=start_utc - span, end=start_utc, scope_user_id=scope
        )
        avg_latency = ai_row.get("avg_latency_ms")

        return UsageOverviewOut(
            ai=UsageAiOverviewOut(
                calls=ai_row["calls"],
                input_tokens=ai_row["input_tokens"],
                output_tokens=ai_row["output_tokens"],
                total_tokens=ai_row["total_tokens"],
                total_cost=ai_row["total_cost"],
                avg_latency_ms=round(avg_latency) if avg_latency is not None else None,
                error_count=ai_row["error_count"],
            ),
            biz=UsageBizOverviewOut(**biz_row),
            compare=UsageCompareOut(
                calls_change_pct=_pct(ai_row["calls"], prev_row["calls"]),
                tokens_change_pct=_pct(ai_row["total_tokens"], prev_row["total_tokens"]),
            ),
            start_time=start_utc,
            end_time=end_utc,
        )

    # ---- 内部 ----
    async def _resolve_scope(self, current: UserContext, only_self: bool) -> UUID | None:
        """解析数据范围；`only_self=True`（个人接口）时强制只看自己。"""
        if only_self:
            if current.user_id is None:
                raise ValidateError("无法识别当前用户")
            return current.user_id
        return await self._scope_user_id(current)


__all__ = ["UsageService", "UsageSource"]
