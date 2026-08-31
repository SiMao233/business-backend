"""AI 能力 Pydantic Schema：请求 / 响应模型。"""

from uuid import UUID

from pydantic import Field, field_serializer

from app.common.schema import ApiInModel, ApiOutModel


class AiChatRequest(ApiInModel):
    """触发 Agent 对话入参。"""

    agent_id: UUID = Field(description="Agent ID")
    input: str = Field(min_length=1, max_length=10000, description="用户输入")
    # 会话 ID（可选；第一版不落库会话，传了也仅用于日志/追踪）
    session_id: str | None = Field(default=None, max_length=64, description="会话 ID（可选）")


class AiChatOut(ApiOutModel):
    """Agent 对话出参。"""

    agent_id: UUID
    reply: str = Field(default="", description="模型回复内容")
    model: str = Field(default="", description="实际使用的模型标识")
    # 是否异步受理（true=已入队后台执行，需轮询结果；false=同步返回 reply）
    async_: bool = Field(default=False, alias="async", description="是否异步受理")
    run_id: str = Field(default="", description="运行 ID（异步时返回，用于查询结果）")

    model_config = {"populate_by_name": True}

    @field_serializer("agent_id")
    def serialize_agent_id(self, value: UUID) -> str:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex
