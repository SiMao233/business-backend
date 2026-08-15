"""通用枚举定义。"""

from enum import StrEnum


class BaseEnum(StrEnum):
    """字符串枚举基类：枚举值可直接作为 JSON 输出。"""

    @classmethod
    def values(cls) -> list[str]:
        """返回全部枚举值列表。"""
        return [item.value for item in cls]


class CommonStatus(BaseEnum):
    """通用启用 / 禁用状态。"""

    ENABLED = "enabled"
    DISABLED = "disabled"
