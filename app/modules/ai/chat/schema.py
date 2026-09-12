"""AI 对话 Pydantic Schema：对话请求 / 响应模型。"""

from uuid import UUID

from pydantic import Field, field_serializer

from app.common.schema import ApiInModel, ApiOutModel


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
    reasoning: str = Field(default="", description="模型推理内容（思维链，推理型模型才有）")
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


# ---- SSE 流式对话事件载荷（配合 chat/api.py 的 /ai/chat 流式端点）----


class AiStreamMetaOut(ApiOutModel):
    """`meta` 事件（首帧）：会话与模型元信息，前端可立即拿到会话 ID。"""

    agent_id: UUID
    session_id: UUID | None = Field(default=None, description="会话 ID（新建时返回）")
    model: str = Field(default="", description="实际使用的模型标识")

    @field_serializer("agent_id", "session_id")
    def serialize_id(self, value: UUID | None) -> str | None:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex if value else None


class AiStreamDeltaOut(ApiOutModel):
    """`delta` 事件：模型增量文本，前端按顺序拼接即为完整回复。"""

    content: str = Field(default="", description="本次增量文本")


class AiStreamReasoningOut(ApiOutModel):
    """`reasoning` 事件：模型思维链增量（仅推理型模型会产生；无则无此事件）。"""

    content: str = Field(default="", description="本次增量推理文本")


class AiStreamToolOut(ApiOutModel):
    """`tool` 事件：检索 / 工具调用过程，便于前端展示「正在检索知识库」等状态。"""

    name: str = Field(description="工具名（knowledge_retrieval / web_search 等）")
    status: str = Field(description="状态：running（开始）/ done（结束）")
    output: str | None = Field(default=None, description="结果摘要（status=done 时，超长会截断）")


class AiStreamErrorOut(ApiOutModel):
    """`error` 事件：流中途错误（此时 HTTP 状态已为 200，只能通过事件告知前端）。"""

    code: int = Field(default=500, description="业务错误码")
    message: str = Field(default="", description="错误提示（可直接展示）")


class AiStreamUsageOut(ApiOutModel):
    """token 用量（模型网关支持时返回，否则为 0）。"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AiStreamDoneOut(ApiOutModel):
    """`done` 事件（尾帧）：完整回复 + 结束原因，前端可用它校正拼接结果。"""

    session_id: UUID | None = Field(default=None, description="会话 ID")
    content: str = Field(default="", description="完整回复内容")
    reasoning: str = Field(default="", description="完整推理内容（可选，推理型模型才有）")
    model: str = Field(default="", description="实际使用的模型标识")
    finish_reason: str = Field(
        default="stop", description="结束原因：stop（正常）/ length（截断）/ error（出错）"
    )
    usage: AiStreamUsageOut | None = Field(default=None, description="token 用量（可选）")

    @field_serializer("session_id")
    def serialize_id(self, value: UUID | None) -> str | None:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex if value else None
