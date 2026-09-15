"""用量统计 Pydantic Schema：请求 / 响应模型。

约定：
- 请求继承 `ApiInModel`，响应继承 `ApiOutModel`（JSON 出入参统一 camelCase）；
- 时间出参使用 `UtcDateTime`（带 Z 的 UTC ISO）；金额（Decimal）序列化为字符串，避免精度丢失。
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated
from uuid import UUID

from pydantic import Field, field_serializer

from app.common.enums import TrendGranularity, UsageCallType, UsageDimension
from app.common.schema import ApiInModel, ApiOutModel, UtcDateTime


# ---- 请求 ----
class UsageRecordQuery(ApiInModel):
    """用量明细分页查询。"""

    page: int = Field(default=1, ge=1, description="页码（从 1 开始）")
    pageSize: int = Field(default=20, ge=1, le=100, description="每页条数")
    user_id: UUID | None = Field(default=None, description="按用户筛选")
    agent_id: UUID | None = Field(default=None, description="按 Agent 筛选")
    model_instance_id: UUID | None = Field(default=None, description="按模型实例筛选")
    organization_id: UUID | None = Field(default=None, description="按组织筛选")
    call_type: UsageCallType | None = Field(default=None, description="调用类型筛选")
    status: int | None = Field(default=None, ge=0, le=1, description="状态筛选：1成功 0失败")
    start_time: datetime | None = Field(default=None, description="开始时间（UTC ISO，含）")
    end_time: datetime | None = Field(default=None, description="结束时间（UTC ISO，不含）")


class UsageSummaryQuery(ApiInModel):
    """用量汇总查询（按维度分组）。"""

    dimension: UsageDimension = Field(
        default=UsageDimension.AGENT, description="聚合维度：user/agent/model/provider/organization"
    )
    top: int = Field(default=20, ge=1, le=100, description="返回条数上限（按 token 降序）")
    start_time: datetime | None = Field(default=None, description="开始时间（UTC ISO，含）")
    end_time: datetime | None = Field(default=None, description="结束时间（UTC ISO，不含）")


class UsageTrendQuery(ApiInModel):
    """用量趋势查询。"""

    granularity: TrendGranularity = Field(default=TrendGranularity.DAY, description="时间粒度")
    start_time: datetime | None = Field(default=None, description="开始时间（UTC ISO，含）")
    end_time: datetime | None = Field(default=None, description="结束时间（UTC ISO，不含）")


# ---- 响应 ----
class UsageRecordOut(ApiOutModel):
    """用量明细出参。"""

    id: UUID
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
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_price: Decimal | None = Field(default=None, description="输入单价（元 / 千 token）")
    output_price: Decimal | None = Field(default=None, description="输出单价（元 / 千 token）")
    total_cost: Decimal = Field(default=Decimal(0), description="成本（元）")
    call_type: str = "chat"
    latency_ms: int | None = None
    status: int = 1
    request_id: str | None = None
    usage_date: date
    create_time: UtcDateTime

    @field_serializer(
        "id",
        "user_id",
        "agent_id",
        "organization_id",
        "model_instance_id",
        "provider_id",
        "conversation_id",
        "message_id",
    )
    def _serialize_uuid(self, value: UUID | None) -> str | None:
        """UUID 输出为不带连字符的 32 位十六进制字符串。"""
        return value.hex if value else None

    @field_serializer("input_price", "output_price", "total_cost")
    def _serialize_decimal(self, value: Decimal | None) -> str | None:
        """金额输出为字符串，避免 JSON 浮点精度问题。"""
        return None if value is None else format(value, "f")


class UsageSummaryItemOut(ApiOutModel):
    """汇总项（按某个维度分组后的一组指标）。"""

    dimension_id: UUID | None = None
    dimension_name: str | None = None
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost: Decimal = Decimal(0)
    error_count: int = 0

    @field_serializer("dimension_id")
    def _serialize_dimension_id(self, value: UUID | None) -> str | None:
        return value.hex if value else None

    @field_serializer("total_cost")
    def _serialize_cost(self, value: Decimal) -> str:
        return format(value, "f")


class UsageTotalOut(ApiOutModel):
    """汇总合计。"""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost: Decimal = Decimal(0)
    error_count: int = 0

    @field_serializer("total_cost")
    def _serialize_cost(self, value: Decimal) -> str:
        return format(value, "f")


class UsageSummaryOut(ApiOutModel):
    """用量汇总出参（list 用 Annotated 声明默认值，避免字段名遮蔽内置 list）。"""

    dimension: str
    list: Annotated[list[UsageSummaryItemOut], Field(default_factory=list)]
    total: UsageTotalOut = Field(default_factory=UsageTotalOut)


class UsageTrendItemOut(ApiOutModel):
    """趋势数据点。"""

    date: str = Field(description="业务自然日（YYYY-MM-DD）")
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost: Decimal = Decimal(0)

    @field_serializer("total_cost")
    def _serialize_cost(self, value: Decimal) -> str:
        return format(value, "f")


class UsageTrendOut(ApiOutModel):
    """用量趋势出参。"""

    granularity: str
    list: Annotated[list[UsageTrendItemOut], Field(default_factory=list)]


class UsageAiOverviewOut(ApiOutModel):
    """AI 用量概览。"""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost: Decimal = Decimal(0)
    avg_latency_ms: int | None = Field(default=None, description="平均延迟（毫秒）")
    error_count: int = 0

    @field_serializer("total_cost")
    def _serialize_cost(self, value: Decimal) -> str:
        return format(value, "f")


class UsageBizOverviewOut(ApiOutModel):
    """业务用量概览（实时统计）。"""

    conversations: int = 0
    messages: int = 0
    documents: int = 0
    chunks: int = 0


class UsageCompareOut(ApiOutModel):
    """与上一等长周期对比（上期为 0 时返回 null）。"""

    calls_change_pct: float | None = None
    tokens_change_pct: float | None = None


class UsageOverviewOut(ApiOutModel):
    """用量概览出参。"""

    ai: UsageAiOverviewOut = Field(default_factory=UsageAiOverviewOut)
    biz: UsageBizOverviewOut = Field(default_factory=UsageBizOverviewOut)
    compare: UsageCompareOut = Field(default_factory=UsageCompareOut)
    start_time: UtcDateTime
    end_time: UtcDateTime
