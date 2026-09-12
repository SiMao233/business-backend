"""AI 会话 Pydantic Schema：会话与消息的请求 / 响应模型。"""

from uuid import UUID

from pydantic import ConfigDict, Field, field_serializer

from app.common.schema import ApiInModel, ApiOutModel, UtcDateTime


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
    reasoning: str | None = None
    create_time: UtcDateTime

    @field_serializer("id", "conversation_id")
    def serialize_id(self, value: UUID) -> str:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex
