"""AI 对话的提示词与消息工具（纯函数：不碰数据库、不依赖会话）。

集中两类无状态逻辑：
- 组装：Agent 配置 / 检索到的参考资料 / 多轮历史 → LLM 消息序列；
- 解析：模型返回的 chunk 或完整消息 → 纯文本、推理内容、token 用量、动作步骤。

流式路径（`stream.py`）与同步路径（`service.py`）共用本模块，保证两条路径口径完全一致
（否则 `done` 事件与历史接口会给出不同的时间线与用量）。

关键约定：
- `REFERENCE_PROMPT` 用 `.format(context=...)` 渲染，**除 `{context}` 外不得出现裸 `{}`**（否则 KeyError）；
  幻觉约束：要求模型只基于资料回答，资料不足/无关时明确拒答（「根据现有资料无法回答该问题」）；
- `collect_steps_from_messages()` 的推理锚点按消息顺序累加
  （`AIMessage` 的 reasoning 先于其后的 `ToolMessage`），
  与流式路径「该步被触发时已产出的推理字符数」同口径。
"""

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.modules.ai.steps import StepKind

# 注入「参考资料」的提示词模板（纯 RAG 模式单独挂 SystemMessage；工具模式拼进 system_prompt）
# 幻觉约束：只允许基于资料回答，资料不足/无关时明确拒答，禁止猜测与编造。
# 注意：模板用 .format() 渲染，除 {context} 外**不得出现裸 `{}`**
REFERENCE_PROMPT = (
    "以下是参考资料，请只基于这些资料回答用户问题。\n"
    "如果资料不足以回答问题，或资料与问题无关，请直接说明"
    "「根据现有资料无法回答该问题」，不要猜测，不要编造资料外的内容。\n"
    "引用资料时请在引用句末尾标注编号，格式为 [n]（n 为资料前的方括号数字）；"
    "未引用资料的句子不要标注，也不要编造不存在的编号。\n\n{context}"
)

# tool 事件结果摘要最大长度（避免把整篇资料推给前端）
TOOL_OUTPUT_LIMIT = 500


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


def chunk_text(chunk: Any) -> str:
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


def chunk_reasoning(chunk: Any) -> str:
    """提取推理增量（ReasoningChatOpenAI 已把它放进 additional_kwargs）。"""
    kwargs = getattr(chunk, "additional_kwargs", None) or {}
    value = kwargs.get("reasoning_content")
    return value if isinstance(value, str) else ""


def accumulate_usage(usage: dict[str, int], chunk: Any, seen_ids: set[str]) -> None:
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
    来源从 `ToolMessage.artifact` 取（knowledge_retrieval 用 content_and_artifact），
    非检索类工具（如 web_search）是纯文本返回，artifact 为 None。
    """
    steps: list[dict] = []
    reasoning_len = 0
    for msg in messages:
        if isinstance(msg, AIMessage):
            reasoning_len += len(chunk_reasoning(msg))
            continue
        if not isinstance(msg, ToolMessage):
            continue
        steps.append(
            {
                "step_id": msg.tool_call_id or msg.name or "tool",
                "kind": StepKind.TOOL.value,
                "name": msg.name or "tool",
                "status": "done",
                "output": chunk_text(msg)[:TOOL_OUTPUT_LIMIT],
                "cost_ms": None,
                "reasoning_offset": reasoning_len,
                "sources": msg.artifact if isinstance(msg.artifact, list) else None,
            }
        )
    return steps
