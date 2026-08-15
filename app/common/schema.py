"""公共 Pydantic 模型基类。"""
from datetime import UTC, datetime
from typing import Annotated

from pydantic import AliasGenerator, BaseModel, ConfigDict, PlainSerializer
from pydantic.alias_generators import to_camel


def _format_utc_iso(dt: datetime) -> str:
    """将时间格式化为带 Z 的 UTC ISO 字符串（如 2026-08-14T04:30:00Z）。

    - ORM 从 MySQL DATETIME 读出的是 naive UTC 墙钟时间，按 UTC 处理；
    - 带时区的对象统一先转 UTC。
    前端可直接 `new Date(value)` 解析，无需手动转时区。
    """
    if dt.tzinfo is None:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# 出参时间类型：统一序列化为带 Z 的 UTC ISO 格式
UtcDateTime = Annotated[
    datetime, PlainSerializer(_format_utc_iso, return_type=str, when_used="json")
]


class ApiOutModel(BaseModel):
    """响应模型基类：输出字段统一转为驼峰。

    - 校验 / ORM 读取仍用字段名（snake_case），model_validate(orm_obj) 不受影响
    - 仅序列化（JSON 输出）用驼峰别名，FastAPI 默认 by_alias=True 自动生效
    - 时间字段请使用 `UtcDateTime` 类型，序列化输出带 Z 的 UTC ISO 字符串
    """

    model_config = ConfigDict(
        from_attributes=True,
        alias_generator=AliasGenerator(serialization_alias=to_camel),
    )


class ApiInModel(BaseModel):
    """请求模型基类：统一将字段名映射为驼峰别名。

    - 代码内使用 snake_case 字段名（符合 PEP 8）
    - JSON 入参统一使用驼峰（alias_generator=to_camel），前端无需传下划线
    - populate_by_name=True 同时兼容 snake_case 传参
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )
