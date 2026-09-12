"""AI 对话 Service 层：对话业务规则与流程编排。

核心职责：加载 Agent 配置 → 解析绑定的模型实例/供应商 → 用 LangChain 构造 LLM →
按 config.tools 走智能体（工具调用）或纯 RAG 检索 → 组装上下文 → 调用模型返回回复。

知识库检索策略：**只要 config.knowledge_ids 非空就先检索**（两种模式一致）。
工具模式下检索本可由模型自主决定，但模型经常不调工具就直接回答，导致「配了知识库却没检索」，
故这里统一预检索并把「参考资料」注入 system prompt；knowledge_retrieval 工具仍保留，供模型追问细节。

两条执行路径：
- 同步 `AiChatService.chat()`：一次拿完整回复（Agent 测试运行 / ARQ 后台任务复用）；
- 流式 `stream_chat(ctx)`：模块级异步生成器，产出 SSE 事件（meta / reasoning / tool / delta / error / done）。

步骤与耗时落库（两条路径结构一致，便于刷新后还原）：
- `ai_message.steps` **只落动作**：`kind=retrieval` 是**后端**预检索（非模型工具调用），
  `kind=tool` 才是模型通过 tool_calls 自主发起的调用；每条带 `reasoning_offset`
  （该步被触发时已产出的推理字符数），且只有 `done` 步骤入库（历史无 running）。
- **出参是「思考 + 动作」合并后的时间线**（`done.steps` 与 `MessageOut.steps`）：
  `thinking` 条目由 `steps.build_timeline()` 派生，只带 `{start, end}` 区间引用 `reasoning` 全文，
  **不落库** → 零迁移、老消息立即获得时间线、切分规则改进后历史消息自动跟随。
- `ai_message.thinking_ms` / `ttft_ms`：思考耗时与首字节耗时，仅流式路径有值
  （同步路径拿不到「首 token」时刻，一律为 NULL）。

设计约束（未来可拆独立 ai-service 的边界）：
- 所有 LLM 调用收敛在本模块，RAG 检索收敛在 retrieval.py，向量库封装在 vector.py；
- 会话与消息的存取统一委托 `ConversationService`（conversation 子包），本模块只管「怎么问模型」；
- 流式生成器**不得使用请求级数据库会话**（HTTP 响应开始后请求会话已归还连接），
  需要检索/落库时用 `AsyncSessionLocal` 开短会话。
"""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import anyio
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import AgentStatus
from app.core.config import get_settings
from app.core.exceptions import BizError, NotFoundError
from app.modules.agent.management.model import Agent
from app.modules.agent.management.repository import AgentRepository, AgentVersionRepository
from app.modules.agent.model.repository import ModelInstanceRepository, ModelProviderRepository
from app.modules.ai.chat.schema import (
    AiChatOut,
    AiStreamDeltaOut,
    AiStreamDoneOut,
    AiStreamErrorOut,
    AiStreamMetaOut,
    AiStreamReasoningOut,
    AiStreamToolOut,
    AiStreamUsageOut,
)
from app.modules.ai.conversation.service import ConversationService
from app.modules.ai.reasoning import ReasoningChatOpenAI
from app.modules.ai.retrieval import format_context, retrieve_hits
from app.modules.ai.steps import AiStepOut, StepKind, build_timeline
from app.modules.ai.tools import build_tools

# 纯 RAG 模式注入「参考资料」的提示词模板
REFERENCE_PROMPT = "以下是参考资料，请优先基于这些资料回答用户问题，不要编造资料外的内容：\n\n{context}"

# tool 事件结果摘要最大长度（避免把整篇资料推给前端）
TOOL_OUTPUT_LIMIT = 500

# 流式结束原因
FINISH_STOP = "stop"
FINISH_ERROR = "error"

# 知识库检索的工具名 / 事件名（预检索与 knowledge_retrieval 工具同名，前端展示统一）
KNOWLEDGE_TOOL = "knowledge_retrieval"


# ---- 内部：纯函数工具 ----


def system_prompt_of(config: dict) -> str:
    """从配置提取 system_prompt。"""
    return str(config.get("system_prompt", "") or "")


def compose_system_prompt(system_prompt: str, context: str) -> str:
    """把 system_prompt 与「参考资料」拼成最终系统提示词（两者都可为空）。

    工具模式（create_agent）只接受一个 system_prompt 字符串，参考资料需拼进这里；
    纯 RAG 模式则用 build_messages 单独挂一条 SystemMessage（见下）。
    """
    if not context:
        return system_prompt
    reference = REFERENCE_PROMPT.format(context=context)
    return f"{system_prompt}\n\n{reference}" if system_prompt else reference


def build_messages(
    system_prompt: str,
    context: str,
    history: list[tuple[str, str]],
    user_input: str,
) -> list[BaseMessage]:
    """组装 LLM 消息：system_prompt →（参考资料）→ 多轮历史 → 当前输入。"""
    messages: list[BaseMessage] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    if context:
        messages.append(SystemMessage(content=REFERENCE_PROMPT.format(context=context)))
    for role, content in history:
        messages.append(
            HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        )
    messages.append(HumanMessage(content=user_input))
    return messages


def _chunk_text(chunk: Any) -> str:
    """从消息/消息块中提取纯文本（兼容 str 与 content blocks 列表）。"""
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict) and block.get("type") in (None, "text"):
                texts.append(str(block.get("text", "")))
        return "".join(texts)
    return ""


def _chunk_reasoning(chunk: Any) -> str:
    """提取推理增量（ReasoningChatOpenAI 已把它放进 additional_kwargs）。"""
    kwargs = getattr(chunk, "additional_kwargs", None) or {}
    value = kwargs.get("reasoning_content")
    return value if isinstance(value, str) else ""


def _accumulate_usage(usage: dict[str, int], chunk: Any, seen_ids: set[str]) -> None:
    """累加 token 用量（按消息 ID 去重，避免同一消息的多个 chunk 重复计数）。"""
    meta = getattr(chunk, "usage_metadata", None)
    if not isinstance(meta, dict):
        return
    chunk_id = getattr(chunk, "id", None)
    if chunk_id and chunk_id in seen_ids:
        return
    if chunk_id:
        seen_ids.add(chunk_id)
    usage["input_tokens"] += int(meta.get("input_tokens") or 0)
    usage["output_tokens"] += int(meta.get("output_tokens") or 0)
    usage["total_tokens"] += int(meta.get("total_tokens") or 0)


def collect_steps_from_messages(messages: list[BaseMessage]) -> list[dict]:
    """同步路径专用：从完整消息序列反推动作步骤与推理锚点（纯函数）。

    非流式拿不到单步耗时，`cost_ms` 一律为 None；无动作时返回空列表。
    锚点按消息顺序累加：`AIMessage` 的 reasoning 先于其后的 `ToolMessage`，
    与流式路径「offset = 该步被触发时已产出的推理字符数」口径一致。
    """
    steps: list[dict] = []
    reasoning_len = 0
    for msg in messages:
        if isinstance(msg, AIMessage):
            reasoning_len += len(_chunk_reasoning(msg))
            continue
        if not isinstance(msg, ToolMessage):
            continue
        steps.append(
            {
                "step_id": msg.tool_call_id or msg.name or "tool",
                "kind": StepKind.TOOL.value,
                "name": msg.name or "tool",
                "status": "done",
                "output": _chunk_text(msg)[:TOOL_OUTPUT_LIMIT],
                "cost_ms": None,
                "reasoning_offset": reasoning_len,
            }
        )
    return steps


# ---- 流式对话的数据结构 ----


@dataclass
class StreamEvent:
    """SSE 事件：`type` 为事件名，`data` 为事件载荷（ApiOutModel）。"""

    type: str
    data: Any


@dataclass
class TokenTimer:
    """耗时打点：分别记录「首个推理增量」与「首个正文增量」的时刻。

    - `thinking_ms`：首个 reasoning → 首个正文（非推理模型无 reasoning → None）
    - `ttft_ms`：服务端进入预检 → 首个正文（含 RAG 检索与装载等待；未产出正文 → None）

    所有 mark 幂等（重复调用不覆盖首值）；`freeze()` 在流结束时调用一次，
    使落库值与 `done` 事件取值完全一致（否则属性里回退 `perf_counter()` 会两次取值不同）。
    """

    started_at: float
    reasoning_at: float | None = None
    text_at: float | None = None
    finished_at: float | None = None

    def mark_reasoning(self) -> None:
        """首个推理增量到达。"""
        if self.reasoning_at is None:
            self.reasoning_at = time.perf_counter()

    def mark_text(self) -> None:
        """首个正文增量到达。"""
        if self.text_at is None:
            self.text_at = time.perf_counter()

    def freeze(self) -> None:
        """流结束（含正常收尾 / 中断 / 报错）时冻结计时。"""
        if self.finished_at is None:
            self.finished_at = time.perf_counter()

    def _thinking_end(self) -> float:
        """思考阶段终点：优先用首个正文；无正文则用冻结时刻。"""
        if self.text_at is not None:
            return self.text_at
        if self.finished_at is not None:
            return self.finished_at
        return time.perf_counter()

    @property
    def thinking_ms(self) -> int | None:
        if self.reasoning_at is None:
            return None
        return max(0, round((self._thinking_end() - self.reasoning_at) * 1000))

    @property
    def ttft_ms(self) -> int | None:
        if self.text_at is None:
            return None
        return max(0, round((self.text_at - self.started_at) * 1000))


@dataclass
class StepCollector:
    """工具 / 检索步骤收集器：统一生成 step_id、单步耗时与推理锚点。

    只记录 `status=done` 的步骤（历史回看不存在 running）。

    **两类 dict 必须分开**（否则 `AiStreamToolOut(**payload)` 会因多键抛 TypeError）：
    - `start()` / `finish()` 的**返回值** = SSE 事件载荷，键名严格等于 `AiStreamToolOut` 字段；
    - `self.steps` 里的**落库结构** = 上面再加 `reasoning_offset`（供 `build_timeline` 定位）。
    """

    steps: list[dict] = field(default_factory=list)
    # step_id -> (起始时刻, 该步被触发时已产出的推理字符数)
    _started: dict[str, tuple[float, int]] = field(default_factory=dict)

    def start(self, step_id: str, name: str, kind: str, reasoning_offset: int = 0) -> dict:
        """登记一次调用的开始，返回可直接推给前端的 running 载荷。"""
        self._started[step_id] = (time.perf_counter(), reasoning_offset)
        return {
            "step_id": step_id,
            "kind": kind,
            "name": name,
            "status": "running",
            "output": None,
            "cost_ms": None,
        }

    def finish(
        self, step_id: str, name: str, kind: str, output: str | None = None
    ) -> dict:
        """登记完成：返回 done 事件载荷，并把带锚点的记录追加进 `self.steps` 供落库。"""
        started = self._started.pop(step_id, None)
        cost_ms = (
            round((time.perf_counter() - started[0]) * 1000) if started is not None else None
        )
        self.steps.append(
            {
                "step_id": step_id,
                "kind": kind,
                "name": name,
                "status": "done",
                "output": output,
                "cost_ms": cost_ms,
                "reasoning_offset": started[1] if started is not None else 0,
            }
        )
        return {
            "step_id": step_id,
            "kind": kind,
            "name": name,
            "status": "done",
            "output": output,
            "cost_ms": cost_ms,
        }

    def dump(self) -> list[dict] | None:
        """落库用：空列表返回 None，避免写入无意义的空 JSON。"""
        return self.steps or None


@dataclass
class ChatStreamContext:
    """流式对话预检上下文（只放纯数据，不持有 ORM 对象与数据库会话）。"""

    agent_id: UUID
    conv_id: UUID | None
    model_code: str
    llm: ChatOpenAI
    config: dict
    history: list[tuple[str, str]]
    user_input: str
    # 请求进入预检的时刻（perf_counter），用于计算 ttft_ms
    started_at: float = 0.0


# ---- 流式执行（模块级：不依赖请求级数据库会话）----


async def _retrieve_hits_short_session(config: dict, query: str) -> list[dict]:
    """用短会话执行 RAG 检索（进入流式阶段后请求级会话已归还连接）。"""
    from app.core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        return await retrieve_hits(db, config, query)


async def _save_reply(
    conv_id: UUID | None,
    reply: str,
    reasoning: str = "",
    thinking_ms: int | None = None,
    ttft_ms: int | None = None,
    steps: list[dict] | None = None,
) -> None:
    """流式结束后落库 assistant 回复（独立短会话；失败只记日志，不影响已推送内容）。

    守卫条件含 `steps`：只检索、未产出正文就被客户端中断时也要落库，
    否则「本次检索过」的记录会丢失（前端刷新后看不到步骤）。
    """
    if conv_id is None or (not reply and not reasoning and not steps):
        return
    # 延迟导入：避免与 core.database 的导入顺序耦合
    from app.core.database import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as db:
            await ConversationService(db).save_assistant_message(
                conv_id, reply, reasoning, thinking_ms, ttft_ms, steps
            )
    except Exception:  # noqa: BLE001 - 落库失败不应影响流式结果
        logger.exception("AI 流式回复落库失败 conversation={}", conv_id)


async def _stream_rag_mode(
    ctx: ChatStreamContext,
    context: str,
    parts: list[str],
    reasoning_parts: list[str],
    usage: dict[str, int],
    timer: TokenTimer,
) -> AsyncIterator[StreamEvent]:
    """纯 RAG 模式：用已检索好的参考资料组装消息，再流式调用模型。"""
    messages = build_messages(system_prompt_of(ctx.config), context, ctx.history, ctx.user_input)
    seen_ids: set[str] = set()
    settings = get_settings()
    async for chunk in ctx.llm.astream(messages, timeout=settings.ai_request_timeout):
        # 推理增量 → reasoning 事件（推理型模型才有；无则静默跳过）
        if reasoning := _chunk_reasoning(chunk):
            timer.mark_reasoning()
            reasoning_parts.append(reasoning)
            yield StreamEvent("reasoning", AiStreamReasoningOut(content=reasoning))
        text = _chunk_text(chunk)
        if text:
            timer.mark_text()
            parts.append(text)
            yield StreamEvent("delta", AiStreamDeltaOut(content=text))
        _accumulate_usage(usage, chunk, seen_ids)


async def _stream_tool_mode(
    ctx: ChatStreamContext,
    tools: list,
    context: str,
    parts: list[str],
    reasoning_parts: list[str],
    usage: dict[str, int],
    timer: TokenTimer,
    collector: StepCollector,
) -> AsyncIterator[StreamEvent]:
    """工具模式：走 LangChain 智能体逐 token 流式，并把工具调用过程推给前端。

    `context` 是预先检索好的参考资料（已推过 tool 事件），注入 system prompt，
    保证「绑定了知识库必定检索」；模型仍可继续调用 knowledge_retrieval 工具追问细节。

    工具步骤用 `tool_call_id` 关联 running/done（而非工具名）：
    流式下 `name` / `id` 只出现在首个 tool_call_chunk，用 name 兜底会漏判同一工具的多次调用。
    """
    from langchain.agents import create_agent

    agent = create_agent(
        ctx.llm, tools, system_prompt=compose_system_prompt(system_prompt_of(ctx.config), context)
    )
    messages = build_messages("", "", ctx.history, ctx.user_input)
    started_tools: set[str] = set()
    seen_ids: set[str] = set()

    # stream_mode="messages"：逐 token 产出 (消息块, 元数据) 二元组
    async for chunk, _meta in agent.astream({"messages": messages}, stream_mode="messages"):
        # 工具执行结果 → tool done（附带结果摘要与该步耗时）
        if isinstance(chunk, ToolMessage):
            step_id = chunk.tool_call_id or chunk.name or "tool"
            yield StreamEvent(
                "tool",
                AiStreamToolOut(
                    **collector.finish(
                        step_id,
                        chunk.name or "tool",
                        StepKind.TOOL.value,
                        _chunk_text(chunk)[:TOOL_OUTPUT_LIMIT],
                    )
                ),
            )
            continue
        # 模型推理增量 → reasoning 事件（推理型模型才有；与正文/tool 可能交错）
        if reasoning := _chunk_reasoning(chunk):
            timer.mark_reasoning()
            reasoning_parts.append(reasoning)
            yield StreamEvent("reasoning", AiStreamReasoningOut(content=reasoning))
        # 模型发起工具调用 → tool running（step_id 即 tool_call_id，可区分同名多次调用）
        for call in getattr(chunk, "tool_call_chunks", None) or []:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            call_id = call.get("id")
            if name and call_id and call_id not in started_tools:
                started_tools.add(call_id)
                yield StreamEvent(
                    "tool",
                    AiStreamToolOut(
                        **collector.start(
                            call_id,
                            name,
                            StepKind.TOOL.value,
                            # 锚点：该步被触发时已产出的推理字符数（决定它插在思考文本的哪个位置）
                            sum(map(len, reasoning_parts)),
                        )
                    ),
                )
        text = _chunk_text(chunk)
        if text:
            timer.mark_text()
            parts.append(text)
            yield StreamEvent("delta", AiStreamDeltaOut(content=text))
        _accumulate_usage(usage, chunk, seen_ids)


async def stream_chat(ctx: ChatStreamContext) -> AsyncIterator[StreamEvent]:
    """流式执行对话，产出 SSE 事件：meta → (reasoning) → (tool) → delta… → done。

    - 流中途异常不再抛出（响应已开始，改不了状态码）：转 `error` 事件后正常收尾；
    - 客户端中断（CancelledError）不发 done，但会把已生成内容落库；
    - 落库用独立短会话 + 取消屏蔽（shield），保证中断时也能写入部分内容；
      只检索、未产出正文就被中断时同样落库（steps 非空即写）；
    - 绑定知识库则先统一检索（两种模式一致），避免工具模式下模型不调工具就"没检索直接答"；
    - 每个事件都带单调递增的 `seq`（由 `_emit` 统一发号，预检索与模型事件共用同一序号空间）；
      实时态前端按 `seq` 交错渲染即可；
    - `done` 事件里的 `steps`（思考 + 动作时间线）/ `thinking_ms` / `ttft_ms`
      与落库值、与历史接口完全一致（先 `freeze()` 再取值，且用同一纯函数）。
    """
    parts: list[str] = []
    reasoning_parts: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    finish_reason = FINISH_STOP
    tools = build_tools(ctx.config)
    timer = TokenTimer(started_at=ctx.started_at)
    collector = StepCollector()
    seq = 0

    def _emit(event: StreamEvent) -> StreamEvent:
        """统一发号：把单调递增的 seq 写进事件载荷并返回。

        计数器**只在这一处**自增：预检索事件与内层生成器事件共用同一个序号空间，
        若分别在两处计数会撞号，前端排序随之错乱。
        """
        nonlocal seq
        seq += 1
        event.data.seq = seq
        return event

    yield _emit(
        StreamEvent(
            "meta",
            AiStreamMetaOut(agent_id=ctx.agent_id, session_id=ctx.conv_id, model=ctx.model_code),
        )
    )

    try:
        # 预检索：只要绑定了知识库就检索（不受"模型是否愿意调工具"影响）
        # kind=retrieval：这是后端主动检索，并非模型发起的工具调用，前端需区别展示
        context = ""
        if ctx.config.get("knowledge_ids"):
            step_id = uuid4().hex
            # 预检索发生在任何 LLM 调用之前，推理必为空 → 锚点恒为 0
            yield _emit(
                StreamEvent(
                    "tool",
                    AiStreamToolOut(
                        **collector.start(step_id, KNOWLEDGE_TOOL, StepKind.RETRIEVAL.value, 0)
                    ),
                )
            )
            hits = await _retrieve_hits_short_session(ctx.config, ctx.user_input)
            context = format_context(hits)
            yield _emit(
                StreamEvent(
                    "tool",
                    AiStreamToolOut(
                        **collector.finish(
                            step_id,
                            KNOWLEDGE_TOOL,
                            StepKind.RETRIEVAL.value,
                            f"命中 {len(hits)} 条参考资料",
                        )
                    ),
                )
            )

        if tools:
            async for event in _stream_tool_mode(
                ctx, tools, context, parts, reasoning_parts, usage, timer, collector
            ):
                yield _emit(event)
        else:
            async for event in _stream_rag_mode(ctx, context, parts, reasoning_parts, usage, timer):
                yield _emit(event)
    except asyncio.CancelledError:
        # 客户端断开：不发 done（连接已断），交由 finally 落库已生成内容
        raise
    except Exception as exc:  # noqa: BLE001 - 统一转 error 事件
        logger.exception("AI 流式对话失败 agent_id={} model={}", ctx.agent_id, ctx.model_code)
        finish_reason = FINISH_ERROR
        yield _emit(
            StreamEvent("error", AiStreamErrorOut(code=500, message=f"模型调用失败: {exc}"))
        )
    finally:
        # 取消屏蔽：客户端中断后仍要完成落库（不 shield 的话 await 会被立即取消）
        with anyio.CancelScope(shield=True):
            # 先冻结计时，保证落库值与 done 事件取值完全一致
            timer.freeze()
            await _save_reply(
                ctx.conv_id,
                "".join(parts),
                "".join(reasoning_parts),
                timer.thinking_ms,
                timer.ttft_ms,
                collector.dump(),
            )

    reasoning_text = "".join(reasoning_parts)
    yield _emit(
        StreamEvent(
            "done",
            AiStreamDoneOut(
                session_id=ctx.conv_id,
                content="".join(parts),
                reasoning=reasoning_text,
                # 时间线：thinking 片段与动作按位置交错（与历史出参用同一纯函数，结果必然一致）
                steps=[
                    AiStepOut.model_validate(s)
                    for s in build_timeline(collector.steps, reasoning_text)
                ],
                thinking_ms=timer.thinking_ms,
                ttft_ms=timer.ttft_ms,
                model=ctx.model_code,
                finish_reason=finish_reason,
                usage=AiStreamUsageOut(**usage),
            ),
        )
    )


class AiChatService:
    """AI 对话服务（同步执行 + 流式预检）。"""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.agent_repo = AgentRepository(db)
        self.version_repo = AgentVersionRepository(db)
        self.instance_repo = ModelInstanceRepository(db)
        self.provider_repo = ModelProviderRepository(db)
        # 会话与消息存取统一委托会话服务（依赖方向单向：chat → conversation）
        self.conv_service = ConversationService(db)

    # ---- 对话（同步）----
    async def chat(
        self,
        agent_id: UUID,
        user_input: str,
        user_id: UUID | None = None,
        session_id: UUID | None = None,
    ) -> AiChatOut:
        """同步执行 Agent 对话：加载配置 → 按 config.tools 走智能体（工具调用）或纯 RAG → 调用模型 → 返回回复。

        - 有 user_id 时自动落库会话（传 session_id 复用，否则新建）；
        - 无 user_id（后台任务）不落库。
        """
        agent, config = await self._load_agent_config(agent_id)
        conv_id, history = await self._resolve_conversation(agent_id, user_input, user_id, session_id)
        llm, model_code = await self._build_llm(agent, config, conv_id)
        system_prompt = system_prompt_of(config)

        # 工具模式：config.tools 配置了工具时走 LangChain 智能体（模型自主决定调工具）
        tools = build_tools(config)
        settings = get_settings()
        reasoning = ""
        steps: list[dict] = []
        try:
            # 预检索：与流式路径保持一致 —— 只要绑定知识库就先检索并注入参考资料，
            # 避免工具模式下模型不调工具就"没检索直接答"；同时记录一条 retrieval 步骤。
            context = ""
            if config.get("knowledge_ids"):
                hits = await retrieve_hits(self.db, config, user_input)
                context = format_context(hits)
                steps.append(
                    {
                        "step_id": uuid4().hex,
                        "kind": StepKind.RETRIEVAL.value,
                        "name": KNOWLEDGE_TOOL,
                        "status": "done",
                        "output": f"命中 {len(hits)} 条参考资料",
                        "cost_ms": None,
                        # 预检索在 LLM 调用之前，推理必为空
                        "reasoning_offset": 0,
                    }
                )

            if tools:
                from langchain.agents import create_agent

                agent_graph = create_agent(
                    llm, tools, system_prompt=compose_system_prompt(system_prompt, context)
                )
                msgs = build_messages("", "", history, user_input)
                result = await agent_graph.ainvoke({"messages": msgs})
                last = result["messages"][-1]
                reply = last.content if isinstance(last.content, str) else str(last.content)
                # 汇总所有 AI 消息的推理内容（工具模式下可能分布在多轮中间消息）
                reasoning = "".join(
                    _chunk_reasoning(m) for m in result["messages"] if isinstance(m, AIMessage)
                )
                # 从完整消息序列反推模型发起的工具调用步骤（非流式无单步耗时）
                steps.extend(collect_steps_from_messages(result["messages"]))
            else:
                # 纯 RAG 模式：组装上下文 → 调用模型
                messages = build_messages(system_prompt, context, history, user_input)
                response = await llm.ainvoke(messages, timeout=settings.ai_request_timeout)
                reply = response.content if hasattr(response, "content") else str(response)
                reasoning = _chunk_reasoning(response)
        except Exception as exc:  # noqa: BLE001 - 统一转业务错误
            logger.exception("LLM 调用失败 agent_id={} model={}", agent_id, model_code)
            raise BizError(f"模型调用失败: {exc}") from exc

        # 落库 assistant 回复（同步路径拿不到「首 token」时刻，两个耗时字段留空）
        if conv_id is not None:
            await self.conv_service.save_assistant_message(
                conv_id, reply, reasoning, None, None, steps or None
            )

        logger.info(
            "AI chat done agent_id={} model={} user_id={} session={}",
            agent_id,
            model_code,
            user_id,
            conv_id,
        )
        return AiChatOut(
            agent_id=agent_id,
            reply=reply,
            reasoning=reasoning,
            # 时间线：thinking 片段与动作按位置交错（与流式路径同一纯函数）
            steps=[AiStepOut.model_validate(s) for s in build_timeline(steps, reasoning)],
            model=model_code,
            async_=False,
            session_id=conv_id,
        )

    # ---- 对话（流式预检）----
    async def prepare_chat(
        self,
        agent_id: UUID,
        user_input: str,
        user_id: UUID | None = None,
        session_id: UUID | None = None,
    ) -> ChatStreamContext:
        """流式对话预检：校验 Agent → 读版本快照 → 解析会话（落库 user 消息）→ 构造 LLM。

        必须在「响应开始前」执行（HTTP 层由依赖调用）：此阶段失败会抛 AppError，
        由全局异常处理器返回统一 JSON 错误体；进入流式阶段后失败只能以 error 事件返回。

        入口处即打点 `started_at`，作为 `ttft_ms`（首字节耗时）的起点。
        """
        started_at = time.perf_counter()
        agent, config = await self._load_agent_config(agent_id)
        conv_id, history = await self._resolve_conversation(agent_id, user_input, user_id, session_id)
        llm, model_code = await self._build_llm(agent, config, conv_id)
        return ChatStreamContext(
            agent_id=agent_id,
            conv_id=conv_id,
            model_code=model_code,
            llm=llm,
            config=config,
            history=history,
            user_input=user_input,
            started_at=started_at,
        )

    # ---- ARQ 任务入口（异步）----
    @staticmethod
    async def chat_task(
        agent_id: str,
        user_input: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        """ARQ 后台任务：独立会话执行对话，返回结果 dict（供 worker 序列化）。"""
        from app.core.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            service = AiChatService(db)
            out = await service.chat(
                UUID(agent_id),
                user_input,
                UUID(user_id) if user_id else None,
                UUID(session_id) if session_id else None,
            )
            return out.model_dump(by_alias=True)

    # ---- 内部：加载 / 校验 ----
    async def _load_agent_config(self, agent_id: UUID) -> tuple[Agent, dict]:
        """加载 Agent 并校验可运行，返回 (agent, 当前版本配置快照)。"""
        agent = await self.agent_repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        if agent.status in (
            AgentStatus.PAUSED.value,
            AgentStatus.STOPPED.value,
            AgentStatus.PENDING.value,
        ):
            raise BizError("Agent 已暂停/停止/待发布，无法运行")
        if not agent.current_version:
            raise BizError("Agent 尚未发布，无法运行")
        # 方案 B：运行时读 current_version 对应的版本快照（发布后生效，修改草稿不影响线上）
        version = await self.version_repo.get_by_version(agent_id, agent.current_version)
        if not version:
            raise BizError("Agent 版本快照不存在")
        return agent, (version.config or {})

    async def _resolve_conversation(
        self,
        agent_id: UUID,
        user_input: str,
        user_id: UUID | None,
        session_id: UUID | None,
    ) -> tuple[UUID | None, list[tuple[str, str]]]:
        """解析会话与多轮历史：有 user_id 时落库（复用/新建），无 user_id（后台任务）不落库。"""
        if user_id is None:
            return session_id, []
        return await self.conv_service.resolve_for_chat(
            agent_id, user_id, session_id, user_input
        )

    # ---- 内部：构造 LLM ----
    async def _build_llm(
        self, agent: Agent, config: dict, session_id: UUID | None = None
    ) -> tuple[ChatOpenAI, str]:
        """根据 Agent 绑定的模型实例构造 ChatOpenAI（OpenAI 兼容协议）。

        返回 (llm, model_code)。Agent 未绑定模型实例时，回退到配置 ai_default_model。
        opencode 网关兼容：base_url 含 opencode 时需携带 x-opencode-session（取会话 ID 前 8 位）。
        """
        settings = get_settings()
        model_code = settings.ai_default_model
        base_url: str | None = None
        api_key: str | None = None

        if agent.model_id is not None:
            instance = await self.instance_repo.get_by_id(agent.model_id)
            if instance is None:
                raise BizError("Agent 绑定的模型实例不存在")
            if instance.status != 1:
                raise BizError("Agent 绑定的模型实例已停用")
            model_code = instance.code
            provider = await self.provider_repo.get_by_id(instance.provider_id)
            if provider is not None:
                base_url = provider.base_url
                api_key = provider.api_key

        if not model_code:
            raise BizError("Agent 未绑定模型实例，且未配置默认模型")

        # opencode 网关兼容：请求需携带 x-opencode-session（取会话 ID 前 8 位）
        default_headers: dict[str, str] | None = None
        if base_url and "opencode" in base_url:
            default_headers = {"x-opencode-session": session_id.hex[:8] if session_id else ""}

        return (
            ReasoningChatOpenAI(
                model=model_code,
                api_key=api_key or "not-set",
                base_url=base_url,
                default_headers=default_headers,
                temperature=float(config.get("temperature", 0.7)),
                max_tokens=int(config.get("max_tokens", 2048)),
                max_retries=settings.ai_max_retries,
                timeout=settings.ai_request_timeout,
                # 流式请求 token 用量（网关不支持时用 AI_STREAM_USAGE=false 关闭）
                stream_usage=settings.ai_stream_usage,
            ),
            model_code,
        )
