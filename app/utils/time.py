"""时间与日期通用工具。"""

from datetime import UTC, datetime


def utcnow() -> datetime:
    """返回带时区的当前 UTC 时间。"""
    return datetime.now(UTC)
