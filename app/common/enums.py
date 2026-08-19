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


class FileBizType(BaseEnum):
    """文件业务类型：标记文件归属的业务场景（通用 file 模块复用）。

    后续新增业务场景时在此扩展枚举值即可。
    """

    KNOWLEDGE_DOC = "knowledge_doc"          # 知识库文档
    AVATAR = "avatar"                        # 用户头像
    AGENT_ATTACHMENT = "agent_attachment"    # Agent 附件 / 图标
    ATTACHMENT = "attachment"                # 通用附件
