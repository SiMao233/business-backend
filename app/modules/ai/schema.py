"""AI 能力 Pydantic Schema：请求 / 响应模型。"""

from uuid import UUID

from pydantic import ConfigDict, Field, field_serializer

from app.common.schema import ApiInModel, ApiOutModel, UtcDateTime


class AiChatRequest(ApiInModel):
    """触发 Agent 对话入参。"""

    agent_id: UUID = Field(description="Agent ID")
    input: str = Field(min_length=1, max_length=10000, description="用户输入")
    # 会话 ID（可选；传了则复用会话并落库多轮记忆，不传则单轮不落库）
    session_id: UUID | None = Field(default=None, description="会话 ID（可选，用于多轮记忆）")


class AiChatOut(ApiOutModel):
    """Agent 对话出参。"""

    agent_id: UUID
    reply: str = Field(default="", description="模型回复内容")
    model: str = Field(default="", description="实际使用的模型标识")
    # 是否异步受理（true=已入队后台执行，需轮询结果；false=同步返回 reply）
    async_: bool = Field(default=False, alias="async", description="是否异步受理")
    run_id: str = Field(default="", description="运行 ID（异步时返回，用于查询结果）")
    # 会话 ID（多轮记忆时返回，前端可保存用于后续对话）
    session_id: UUID | None = Field(default=None, description="会话 ID（多轮记忆时返回）")

    model_config = {"populate_by_name": True}

    @field_serializer("agent_id", "session_id")
    def serialize_id(self, value: UUID | None) -> str | None:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex if value else None


class ConversationCreate(ApiInModel):
    """创建会话入参。"""

    agent_id: UUID = Field(description="Agent ID")
    title: str | None = Field(default=None, max_length=255, description="会话标题（默认取首条消息）")


class ConversationOut(ApiOutModel):
    """会话出参。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    agent_id: UUID
    user_id: UUID | None = None
    title: str
    status: str
    create_time: UtcDateTime
    update_time: UtcDateTime

    @field_serializer("id", "agent_id", "user_id")
    def serialize_id(self, value: UUID | None) -> str | None:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex if value else None


class ConversationQuery(ApiInModel):
    """会话分页查询。"""

    page: int = Field(default=1, ge=1, description="页码（从 1 开始）")
    pageSize: int = Field(default=20, ge=10, le=100, description="每页条数")
    agent_id: UUID | None = Field(default=None, description="按 Agent 筛选")


class MessageOut(ApiOutModel):
    """会话消息出参。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    conversation_id: UUID
    role: str
    content: str
    create_time: UtcDateTime

    @field_serializer("id", "conversation_id")
    def serialize_id(self, value: UUID) -> str:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex
