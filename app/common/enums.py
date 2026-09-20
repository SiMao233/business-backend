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


class ModelType(BaseEnum):
    """模型实例类型。"""

    CHAT = "chat"           # 对话 / 文本生成
    EMBEDDING = "embedding" # 向量化
    RERANK = "rerank"       # 重排序（RAG 精排）
    IMAGE = "image"         # 图像生成
    AUDIO = "audio"         # 语音


class AgentStatus(BaseEnum):
    """Agent 生命周期状态。

    - draft：草稿，可编辑、不可运行（current_version 为空时线上继续跑旧版本）
    - pending：待发布（已修改配置，需重新发布）
    - running：运行中，可运行、不可编辑
    - paused：暂停，不可运行、可恢复
    - stopped：停止，不可运行
    """

    DRAFT = "draft"         # 草稿（可编辑）
    PENDING = "pending"     # 待发布（已修改配置，需重新发布）
    RUNNING = "running"     # 运行中（可运行）
    PAUSED = "paused"       # 暂停（可恢复）
    STOPPED = "stopped"     # 停止


class AgentVersionStatus(BaseEnum):
    """Agent 版本状态。"""

    DRAFT = "draft"         # 草稿（未发布）
    PUBLISHED = "published" # 已发布（不可修改）


class DocumentStatus(BaseEnum):
    """知识库文档处理状态（切分/向量化流程）。

    - pending：已登记，待处理
    - parsing：切分/向量化中（ARQ 后台任务）
    - parsed：已解析并向量化完成
    - failed：处理失败（error_message 记录原因）
    """

    PENDING = "pending"     # 待处理
    PARSING = "parsing"     # 处理中
    PARSED = "parsed"       # 已解析
    FAILED = "failed"       # 处理失败


class ChunkType(BaseEnum):
    """知识库切分块的类型（结构感知切分产出）。

    - text：普通正文（段落 / 列表，可含标题行）
    - table：整块表格（未与正文混合）
    - mixed：表格与正文被归入同一块
    """

    TEXT = "text"       # 正文
    TABLE = "table"     # 表格
    MIXED = "mixed"     # 正文 + 表格


class UsageCallType(BaseEnum):
    """用量记录对应的模型调用类型。"""

    CHAT = "chat"             # 对话（纯 RAG / 普通对话）
    TOOL = "tool"             # 工具（智能体）模式
    EMBEDDING = "embedding"   # 向量化（第二期接入）


class UsageDimension(BaseEnum):
    """用量统计聚合维度。"""

    USER = "user"                   # 按用户
    AGENT = "agent"                 # 按 Agent
    MODEL = "model"                 # 按模型实例
    PROVIDER = "provider"           # 按模型供应商
    ORGANIZATION = "organization"   # 按组织（取自 Agent.organization_id）


class TrendGranularity(BaseEnum):
    """用量趋势时间粒度。"""

    DAY = "day"     # 按业务自然日
    HOUR = "hour"   # 按小时（暂未支持）
