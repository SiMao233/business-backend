"""用量统计工具：业务自然日、成本计算、查询时间区间规范化。"""

from datetime import UTC, date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from zoneinfo import ZoneInfo

from app.core.config import get_settings
from app.core.exceptions import ValidateError

# 单次查询最大时间跨度（天）：防止全表扫描
MAX_RANGE_DAYS = 366
# 默认查询区间（天）
DEFAULT_RANGE_DAYS = 7
# 单价计价单位：元 / 千 token（与模型实例 input_price / output_price 口径一致）
PRICE_UNIT_TOKENS = Decimal(1000)
# 金额精度：与 ai_usage_record.total_cost 的 Numeric(18, 6) 对齐
_COST_QUANT = Decimal("0.000001")


def business_date(now: datetime | None = None) -> date:
    """按业务时区（settings.usage_timezone）把时刻切为自然日。

    写入用量明细时调用，落到 `ai_usage_record.usage_date`，供按天聚合使用，
    避免运行时 `CONVERT_TZ` 依赖 MySQL 时区表。
    """
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(ZoneInfo(get_settings().usage_timezone)).date()


def calc_cost(
    input_tokens: int,
    output_tokens: int,
    input_price: Decimal | None,
    output_price: Decimal | None,
) -> Decimal:
    """按单价（元 / 千 token）计算单次调用成本；单价为空按 0 计。"""
    in_price = Decimal(input_price or 0)
    out_price = Decimal(output_price or 0)
    cost = (
        Decimal(input_tokens) / PRICE_UNIT_TOKENS * in_price
        + Decimal(output_tokens) / PRICE_UNIT_TOKENS * out_price
    )
    return cost.quantize(_COST_QUANT, rounding=ROUND_HALF_UP)


def _to_utc(moment: datetime) -> datetime:
    """转为 UTC；naive 时间按 UTC 处理（与库内存储口径一致）。"""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def normalize_range(
    start: datetime | None,
    end: datetime | None,
    *,
    default_days: int = DEFAULT_RANGE_DAYS,
) -> tuple[datetime, datetime]:
    """规范化查询时间区间：默认近 N 天、统一 UTC、校验先后顺序与最大跨度。"""
    now = datetime.now(UTC)
    end_utc = _to_utc(end) if end is not None else now
    start_utc = _to_utc(start) if start is not None else end_utc - timedelta(days=default_days)
    if start_utc >= end_utc:
        raise ValidateError("开始时间必须早于结束时间")
    if end_utc - start_utc > timedelta(days=MAX_RANGE_DAYS):
        raise ValidateError(f"查询时间跨度不能超过 {MAX_RANGE_DAYS} 天")
    return start_utc, end_utc
