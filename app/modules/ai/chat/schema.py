"""AI 对话 Pydantic Schema：对话请求 / 响应模型。"""

from uuid import UUID

from pydantic import Field, field_serializer

from app.common.schema import ApiInModel, ApiOutModel
from app.modules.ai.steps import AiSourceOut, AiStepOut, StepKind


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
    # 工具 / 检索步骤（与历史出参 MessageOut.steps 同构，便于刷新后一致展示）
    steps: list[AiStepOut] = Field(default_factory=list, description="检索 / 工具调用步骤")
    thinking_ms: int | None = Field(
        default=None, description="思考耗时（毫秒；首个 reasoning → 首个正文，无推理内容时为 null）"
    )
    ttft_ms: int | None = Field(
        default=None, description="首字节耗时（毫秒；服务端预检 → 首个正文，未产出正文时为 null）"
    )
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


class AiStreamEventOut(ApiOutModel):
    """SSE 事件基类：所有事件统一携带单调递增序号，供前端交错渲染时间线。

    `seq` 从 1 开始、在整个流内单调递增（预检索事件与模型事件共用同一计数器）；
    流内的实际次序以 `seq` 为准，不要依赖事件到达的相对时序。
    **不落库**：历史回看以 `MessageOut.steps` 的数组顺序为准。
    """

    seq: int = Field(default=0, description="事件序号（从 1 单调递增，用于还原事件先后顺序）")


class AiStreamMetaOut(AiStreamEventOut):
    """`meta` 事件（首帧）：会话与模型元信息，前端可立即拿到会话 ID。"""

    agent_id: UUID
    session_id: UUID | None = Field(default=None, description="会话 ID（新建时返回）")
    model: str = Field(default="", description="实际使用的模型标识")

    @field_serializer("agent_id", "session_id")
    def serialize_id(self, value: UUID | None) -> str | None:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex if value else None


class AiStreamDeltaOut(AiStreamEventOut):
    """`delta` 事件：模型增量文本，前端按顺序拼接即为完整回复。"""

    content: str = Field(default="", description="本次增量文本")


class AiStreamReasoningOut(AiStreamEventOut):
    """`reasoning` 事件：模型思维链增量（仅推理型模型会产生；无则无此事件）。"""

    content: str = Field(default="", description="本次增量推理文本")


class AiStreamToolOut(AiStreamEventOut):
    """`tool` 事件：检索 / 工具调用过程，便于前端展示「正在检索知识库」等状态。

    `status=done` 且为知识库检索时带 `sources`，因此来源卡片无需等到 `done` 事件即可渲染。
    """

    step_id: str = Field(default="", description="步骤唯一 ID（与历史 steps[].stepId 一致）")
    kind: str = Field(
        default=StepKind.TOOL.value,
        description="步骤类型：retrieval（后端预检索）/ tool（模型发起的工具调用）",
    )
    name: str = Field(description="工具名（knowledge_retrieval / web_search 等）")
    status: str = Field(description="状态：running（开始）/ done（结束）")
    output: str | None = Field(default=None, description="结果摘要（status=done 时，超长会截断）")
    cost_ms: int | None = Field(default=None, description="该步耗时（毫秒，仅 status=done 时有值）")
    sources: list[AiSourceOut] | None = Field(
        default=None,
        description=(
            "检索来源列表（仅 status=done 且为知识库检索时有值，每 chunk 一条；"
            "index 与正文角标 [n] 对应）"
        ),
    )


class AiStreamErrorOut(AiStreamEventOut):
    """`error` 事件：流中途错误（此时 HTTP 状态已为 200，只能通过事件告知前端）。"""

    code: int = Field(default=500, description="业务错误码")
    message: str = Field(default="", description="错误提示（可直接展示）")


class AiStreamUsageOut(ApiOutModel):
    """token 用量（模型网关支持时返回，否则为 0）。"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AiStreamDoneOut(AiStreamEventOut):
    """`done` 事件（尾帧）：完整回复 + 结束原因，前端可用它校正拼接结果。

    `steps` 是**思考 + 动作**合并后的时间线（含 `kind=thinking` 片段），
    与刷新后 `GET /messages` 的 `steps` 由同一纯函数生成，两者完全一致。
    """

    session_id: UUID | None = Field(default=None, description="会话 ID")
    content: str = Field(default="", description="完整回复内容")
    reasoning: str = Field(default="", description="完整推理内容（可选，推理型模型才有）")
    steps: list[AiStepOut] = Field(
        default_factory=list, description="思考 / 检索 / 工具统一时间线（thinking 条目引用 reasoning 区间）"
    )
    thinking_ms: int | None = Field(
        default=None, description="思考耗时（毫秒；无推理内容时为 null）"
    )
    ttft_ms: int | None = Field(default=None, description="首字节耗时（毫秒；未产出正文时为 null）")
    model: str = Field(default="", description="实际使用的模型标识")
    finish_reason: str = Field(
        default="stop", description="结束原因：stop（正常）/ length（截断）/ error（出错）"
    )
    usage: AiStreamUsageOut | None = Field(default=None, description="token 用量（可选）")

    @field_serializer("session_id")
    def serialize_id(self, value: UUID | None) -> str | None:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex if value else None
