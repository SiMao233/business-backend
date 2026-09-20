"""AI 对话的流式执行：SSE 事件产出、工具/检索步骤收集与旁路落库。

`stream_chat(ctx)` 是**模块级**异步生成器，供 `chat/api.py` 的 SSE 端点直接消费。
从预检上下文（`runtime.ChatStreamContext`）出发，不依赖请求级数据库会话：

- 检索：`_retrieve_hits_short_session()` 自开 `AsyncSessionLocal` 短会话；
- 落库：`_save_reply()` / `persist_usage()` 同样用短会话，且属**旁路能力**，
  失败只记日志，绝不影响已推送的内容。

执行流程：`meta` →（预检索 `tool`）→ 证据不足则 ABSTAIN（后端固定文案，**不调用 LLM**），
否则按 `config.tools` 走工具模式或纯 RAG → `reasoning` / `delta` 增量 → `done`
（带步骤时间线、耗时、用量与 `grounding` 引用审计）。
流中途异常转 `error` 事件（HTTP 仍 200）；客户端中断（`CancelledError`）不发 `done`，
但会在 `finally` 里屏蔽取消、落库已生成内容。
每个事件都带单调递增的 `seq`（由 `_emit` 统一发号，预检索与模型事件共用同一序号空间）。

口径一致性（与同步路径 `service.py` 必须一致，否则 `done` 与历史接口对不上）：
- 步骤只落 `done`（`StepCollector`），每条带 `reasoning_offset`；
- `done` 里的 `steps` 是 `steps.build_timeline()` 派生的「思考 + 动作」时间线；
- `thinking_ms` / `ttft_ms` 先 `timer.freeze()` 再取值。
"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import anyio
from langchain_core.messages import ToolMessage
from loguru import logger

from app.common.enums import UsageCallType
from app.core.config import get_settings
from app.modules.ai.chat.messages import (
    TOOL_OUTPUT_LIMIT,
    accumulate_usage,
    build_messages,
    chunk_reasoning,
    chunk_text,
    compose_system_prompt,
    system_prompt_of,
)
from app.modules.ai.chat.runtime import (
    FINISH_ABSTAIN,
    FINISH_ERROR,
    FINISH_STOP,
    KNOWLEDGE_TOOL,
    ChatStreamContext,
    StepCollector,
    StreamEvent,
    TokenTimer,
)
from app.modules.ai.chat.schema import (
    AiStreamDeltaOut,
    AiStreamDoneOut,
    AiStreamErrorOut,
    AiStreamMetaOut,
    AiStreamReasoningOut,
    AiStreamToolOut,
    AiStreamUsageOut,
)
from app.modules.ai.conversation.service import ConversationService
from app.modules.ai.grounding import (
    EvidenceAssessment,
    abstain_message,
    assess_evidence,
    collect_sources,
    grounding_out,
    grounding_prompt_enabled,
    needs_no_evidence_note,
    should_abstain,
)
from app.modules.ai.query import apply_query_rewrite
from app.modules.ai.retrieval import build_sources, format_context, retrieve_hits
from app.modules.ai.steps import AiStepOut, StepKind, build_timeline
from app.modules.ai.tools import build_tools

# ---- 旁路能力：独立短会话落库（不得影响流式结果）----


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
) -> UUID | None:
    """流式结束后落库 assistant 回复（独立短会话；失败只记日志，不影响已推送内容）。

    守卫条件含 `steps`：只检索、未产出正文就被客户端中断时也要落库，
    否则「本次检索过」的记录会丢失（前端刷新后看不到步骤）。

    返回落库消息的 ID（供用量明细关联 `message_id`）；未落库或失败时返回 None。
    """
    if conv_id is None or (not reply and not reasoning and not steps):
        return None
    # 延迟导入：避免与 core.database 的导入顺序耦合
    from app.core.database import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as db:
            message = await ConversationService(db).save_assistant_message(
                conv_id, reply, reasoning, thinking_ms, ttft_ms, steps
            )
            return message.id if message else None
    except Exception:  # noqa: BLE001 - 落库失败不应影响流式结果
        logger.exception("AI 流式回复落库失败 conversation={}", conv_id)
        return None


async def persist_usage(**fields: Any) -> None:
    """用独立短会话写入一条用量明细（失败只记日志，绝不影响对话主流程）。

    统一出口：流式路径（`_record_usage`）与同步路径（`AiChatService.chat`）都经此落库，
    避免「请求级会话已归还 / 已提交」的时序问题。
    """
    from app.core.database import AsyncSessionLocal
    from app.modules.usage.service import UsageService, UsageSource

    try:
        async with AsyncSessionLocal() as db:
            await UsageService(db).record(UsageSource(**fields))
    except Exception:  # noqa: BLE001 - 旁路能力，落库失败不得影响对话
        logger.exception("AI 用量落库失败 agent_id={}", fields.get("agent_id"))


async def _record_usage(
    ctx: ChatStreamContext,
    usage: dict[str, int],
    *,
    message_id: UUID | None = None,
    latency_ms: int | None = None,
    status: int = 1,
) -> None:
    """流式路径：按上下文快照落库用量。

    刻意与 `_save_reply` 分离：即使没有会话（ARQ 后台任务 / Agent 测试运行）或没有正文，
    只要发生了模型调用就应记录用量，因此**不能**放进 `_save_reply` 的守卫分支。
    """
    await persist_usage(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        user_id=ctx.user_id,
        username=ctx.username,
        agent_id=ctx.agent_id,
        agent_name=ctx.agent_name,
        organization_id=ctx.organization_id,
        model_instance_id=ctx.model_instance_id,
        provider_id=ctx.provider_id,
        provider_code=ctx.provider_code,
        model_code=ctx.model_code,
        conversation_id=ctx.conv_id,
        message_id=message_id,
        input_price=ctx.input_price,
        output_price=ctx.output_price,
        call_type=(
            UsageCallType.TOOL.value if ctx.config.get("tools") else UsageCallType.CHAT.value
        ),
        latency_ms=latency_ms,
        status=status,
    )


# ---- 两条执行路径的内层生成器 ----


async def _stream_rag_mode(
    ctx: ChatStreamContext,
    context: str,
    parts: list[str],
    reasoning_parts: list[str],
    usage: dict[str, int],
    timer: TokenTimer,
    *,
    grounding: bool,
    no_evidence_note: bool,
) -> AsyncIterator[StreamEvent]:
    """纯 RAG 模式：用已检索好的参考资料组装消息，再流式调用模型。

    `grounding` / `no_evidence_note` 由 `stream_chat` 从 grounding.py 的判定结果传入，
    决定是否注入严格接地规则与「本轮无资料」说明（两者都交给 `build_messages`）。
    """
    messages = build_messages(
        system_prompt_of(ctx.config),
        context,
        ctx.history,
        ctx.user_input,
        grounding=grounding,
        no_evidence_note=no_evidence_note,
    )
    seen_ids: set[str] = set()
    settings = get_settings()
    async for chunk in ctx.llm.astream(messages, timeout=settings.ai_request_timeout):
        # 推理增量 → reasoning 事件（推理型模型才有；无则静默跳过）
        if reasoning := chunk_reasoning(chunk):
            timer.mark_reasoning()
            reasoning_parts.append(reasoning)
            yield StreamEvent("reasoning", AiStreamReasoningOut(content=reasoning))
        text = chunk_text(chunk)
        if text:
            timer.mark_text()
            parts.append(text)
            yield StreamEvent("delta", AiStreamDeltaOut(content=text))
        accumulate_usage(usage, chunk, seen_ids)


async def _stream_tool_mode(
    ctx: ChatStreamContext,
    tools: list,
    context: str,
    parts: list[str],
    reasoning_parts: list[str],
    usage: dict[str, int],
    timer: TokenTimer,
    collector: StepCollector,
    *,
    grounding: bool,
    no_evidence_note: bool,
) -> AsyncIterator[StreamEvent]:
    """工具模式：走 LangChain 智能体逐 token 流式，并把工具调用过程推给前端。

    `context` 是预先检索好的参考资料（已推过 tool 事件），注入 system prompt，
    保证「绑定了知识库必定检索」；模型仍可继续调用 knowledge_retrieval 工具追问细节。

    工具步骤用 `tool_call_id` 关联 running/done（而非工具名）：
    流式下 `name` / `id` 只出现在首个 tool_call_chunk，用 name 兜底会漏判同一工具的多次调用。
    """
    from langchain.agents import create_agent

    agent = create_agent(
        ctx.llm,
        tools,
        system_prompt=compose_system_prompt(
            system_prompt_of(ctx.config),
            context,
            grounding=grounding,
            no_evidence_note=no_evidence_note,
        ),
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
                        chunk_text(chunk)[:TOOL_OUTPUT_LIMIT],
                        # 知识库检索工具的结构化来源放在 artifact（不进模型上下文）
                        chunk.artifact if isinstance(chunk.artifact, list) else None,
                    )
                ),
            )
            continue
        # 模型推理增量 → reasoning 事件（推理型模型才有；与正文/tool 可能交错）
        if reasoning := chunk_reasoning(chunk):
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
        text = chunk_text(chunk)
        if text:
            timer.mark_text()
            parts.append(text)
            yield StreamEvent("delta", AiStreamDeltaOut(content=text))
        accumulate_usage(usage, chunk, seen_ids)


# ---- 流式主流程 ----


async def stream_chat(ctx: ChatStreamContext) -> AsyncIterator[StreamEvent]:
    """流式执行对话，产出 SSE 事件：meta → (reasoning) → (tool) → delta… → done。

    - 流中途异常不再抛出（响应已开始，改不了状态码）：转 `error` 事件后正常收尾；
    - 客户端中断（CancelledError）不发 done，但会把已生成内容落库；
    - 落库用独立短会话 + 取消屏蔽（shield），保证中断时也能写入部分内容；
      只检索、未产出正文就被中断时同样落库（steps 非空即写）；
    - 绑定知识库则先统一检索（两种模式一致），避免工具模式下模型不调工具就"没检索直接答"；
    - 预检索后先做证据判定（`grounding.py`）：证据不足且未配置非知识库工具时进入 ABSTAIN ——
      直接返回固定拒答文案、**不调用 Chat LLM**（`finish_reason=abstain`）；
      拒答开关关闭 / 配了其它工具时改为注入 `NO_EVIDENCE_PROMPT` 强约束（概率性）；
    - 检索前可选执行 Query Rewrite（`RAG_QUERY_REWRITE_ENABLED`）：它只替换**检索用 query**，
      发给模型的仍是原始 `ctx.user_input`；失败 / 超时 / 输出非法则自动回退原问题；
    - 每个事件都带单调递增的 `seq`（由 `_emit` 统一发号，预检索与模型事件共用同一序号空间）；
      实时态前端按 `seq` 交错渲染即可；
    - `done` 事件里的 `steps`（思考 + 动作时间线）/ `thinking_ms` / `ttft_ms`
      与落库值、与历史接口完全一致（先 `freeze()` 再取值，且用同一纯函数）。
    """
    parts: list[str] = []
    reasoning_parts: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    finish_reason = FINISH_STOP
    # 证据接地判定（未绑定知识库时保持 None，grounding_out 按「无法判定」处理）
    assessment: EvidenceAssessment | None = None
    abstain = False
    no_evidence_note = False
    # 严格接地规则：绑定知识库且总开关开启才注入（开关只读这一处）
    grounding = grounding_prompt_enabled(ctx.config)
    # 引用编号计数器：预检索与模型主动调用的检索工具共用，保证 [n] 在单次请求内全局唯一
    cited: dict[str, int] = {"n": 0}
    tools = build_tools(ctx.config, cited)
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
            # 查询改写（可选）：放在 collector.start 之后，耗时会计入该检索步骤的 cost_ms。
            # ⚠️ 只影响检索用 query —— 下面 build_messages 仍用 ctx.user_input（原始问题），
            # 改写结果绝不替换发给模型的消息，也绝不进入最终答案。
            rewrite = await apply_query_rewrite(ctx.rewrite_llm, ctx.user_input, ctx.history)
            hits = await _retrieve_hits_short_session(ctx.config, rewrite.rewritten_query)
            # 证据接地判定：与同步路径共用 grounding.py 的同一套函数，口径必然一致
            assessment = assess_evidence(hits)
            abstain = should_abstain(hits, ctx.config)
            no_evidence_note = needs_no_evidence_note(hits, ctx.config)
            # 读改写必须紧邻（中间不得有 await）：与工具的并发调用也不会撞号
            start = cited["n"]
            cited["n"] += len(hits)
            context = format_context(hits, start)
            # 拒答时把判定结论写进步骤摘要：前端不解析 grounding 出参也能看懂发生了什么
            summary = f"命中 {len(hits)} 条参考资料"
            if abstain:
                summary = (
                    f"{summary}（相关度不足，已拒答）" if hits else "未命中参考资料（已拒答）"
                )
            yield _emit(
                StreamEvent(
                    "tool",
                    AiStreamToolOut(
                        **collector.finish(
                            step_id,
                            KNOWLEDGE_TOOL,
                            StepKind.RETRIEVAL.value,
                            summary,
                            build_sources(hits, start) or None,
                        )
                    ),
                )
            )

        if abstain:
            # 证据不足：直接返回后端固定文案，**不调用任何 LLM**（唯一确定性的「不猜」保证）。
            # 仍推 delta + done，保持前端「按增量拼接正文」的既有契约不变。
            answer = abstain_message()
            timer.mark_text()
            parts.append(answer)
            finish_reason = FINISH_ABSTAIN
            yield _emit(StreamEvent("delta", AiStreamDeltaOut(content=answer)))
        elif tools:
            async for event in _stream_tool_mode(
                ctx,
                tools,
                context,
                parts,
                reasoning_parts,
                usage,
                timer,
                collector,
                grounding=grounding,
                no_evidence_note=no_evidence_note,
            ):
                yield _emit(event)
        else:
            async for event in _stream_rag_mode(
                ctx,
                context,
                parts,
                reasoning_parts,
                usage,
                timer,
                grounding=grounding,
                no_evidence_note=no_evidence_note,
            ):
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
            message_id = await _save_reply(
                ctx.conv_id,
                "".join(parts),
                "".join(reasoning_parts),
                timer.thinking_ms,
                timer.ttft_ms,
                collector.dump(),
            )
            # 用量独立落库：即使无会话 / 无正文也要记录，故不放进 _save_reply 的守卫分支
            await _record_usage(
                ctx,
                usage,
                message_id=message_id,
                latency_ms=round((time.perf_counter() - ctx.started_at) * 1000),
                status=0 if finish_reason == FINISH_ERROR else 1,
            )

    reasoning_text = "".join(reasoning_parts)
    answer_text = "".join(parts)
    yield _emit(
        StreamEvent(
            "done",
            AiStreamDoneOut(
                session_id=ctx.conv_id,
                content=answer_text,
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
                grounding=grounding_out(
                    assessment, collect_sources(collector.steps), answer_text, abstained=abstain
                ),
            ),
        )
    )
