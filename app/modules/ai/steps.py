"""AI 对话「思考 / 检索 / 工具」统一时间线的共享 Schema 与纯函数。

放在 ai 根目录而非 chat/schema.py：chat 与 conversation 两个子包都要用（出参同构），
而现有依赖方向是 chat → conversation，若放 chat 再由 conversation 反向 import 会形成模块环。

`kind` 语义（三者混排在同一条 `steps` 时间线里，务必与前端对齐）：
- `thinking`：模型的思考文本片段，**只带 {start, end} 区间**，文本本体在同消息的 `reasoning` 字段里；
- `retrieval`：**后端**在进入模型前无条件执行的预检索（读了 config.knowledge_ids），
  它不是模型发起的工具调用，UI 上不应表述为「模型调用了工具」；
- `tool`：**模型**通过 tool_calls 自主发起的工具调用（knowledge_retrieval / web_search）。

存储分层（重要）：DB 的 `ai_message.steps` **只存动作**（retrieval / tool，各带 `reasoning_offset` 锚点）；
`thinking` 条目**不落库**，由 `build_timeline()` 在读时（或流式 done 时）派生。好处：
零迁移、老消息立即获得时间线、切分规则改进后历史消息自动跟随。

⚠️ 契约影响：`steps` 是「思考 + 动作」的混合时间线，因此
- `steps.length` **不等于**动作数，统计动作要 `filter(kind != "thinking")`；
- 渲染 thinking 必须同时取同消息的 `reasoning` 字段（`reasoning[start:end]`）。
"""

import re
from enum import StrEnum

from pydantic import Field

from app.common.schema import ApiOutModel


class StepKind(StrEnum):
    """步骤类型：思考片段 / 后端预检索 / 模型发起的工具调用。"""

    THINKING = "thinking"
    RETRIEVAL = "retrieval"
    TOOL = "tool"


class AiStepOut(ApiOutModel):
    """时间线中的一条：思考片段（thinking）或一次动作调用（retrieval / tool）。

    平铺超集：不同 `kind` 下有意义的字段不同，**前端必须按 `kind` 分支**：
    - `thinking`：只读 `start` / `end`（配合同消息的 `reasoning` 字段做 slice），其余为 null；
    - `retrieval` / `tool`：只读 `name` / `output` / `cost_ms` / `reasoning_offset`，
      `start` / `end` 为 null。
    """

    step_id: str = Field(
        description=(
            "步骤唯一 ID。动作取 tool_call_id（预检索为服务端生成值）；"
            "thinking 为基于位置的**确定性** ID（每次读取结果一致，不能是随机值）"
        )
    )
    kind: str = Field(
        default=StepKind.TOOL.value,
        description="thinking（思考片段）/ retrieval（后端预检索）/ tool（模型发起的工具调用）",
    )
    status: str = Field(default="done", description="状态：固定 done（历史数据不存在 running）")

    # ---- 动作（retrieval / tool）专用 ----
    name: str = Field(default="", description="工具名；thinking 条目为空串")
    output: str | None = Field(default=None, description="结果摘要（超长已截断）")
    cost_ms: int | None = Field(default=None, description="该步耗时（毫秒）")
    reasoning_offset: int | None = Field(
        default=None,
        description="该动作被触发时已产出的推理字符数；thinking 条目为 null（老数据缺失按 0 处理）",
    )

    # ---- thinking 专用 ----
    start: int | None = Field(default=None, description="思考片段在 reasoning 中的起始下标")
    end: int | None = Field(default=None, description="思考片段在 reasoning 中的结束下标（不含）")


# ---- 纯函数：思考分段 + 时间线合并 ----
# 编号列表行（如 "1. xxx" / "2、xxx" / "3) xxx"）
_ORDERED_LINE = re.compile(r"^\s*\d+[.、)]\s*")
# 空行分隔（段落边界，允许行内有空白字符）
_BLANK_SEPARATOR = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """去掉 [start, end) 首尾空白，返回新区间。"""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _blank_blocks(text: str) -> list[tuple[int, int]]:
    """按空行切块，返回每块去首尾空白后的 (start, end)，丢弃空块。"""
    raw: list[tuple[int, int]] = []
    pos = 0
    for match in _BLANK_SEPARATOR.finditer(text):
        raw.append((pos, match.start()))
        pos = match.end()
    raw.append((pos, len(text)))

    blocks: list[tuple[int, int]] = []
    for start, end in raw:
        start, end = _trim(text, start, end)
        if end > start:
            blocks.append((start, end))
    return blocks


def _content_lines(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """块内所有非空行去首尾空白后的 (start, end)。"""
    lines: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        newline = text.find("\n", pos, end)
        line_end = end if newline < 0 else newline
        line_start, line_end = _trim(text, pos, line_end)
        if line_end > line_start:
            lines.append((line_start, line_end))
        pos = line_end + 1
    return lines


def split_reasoning_segments(text: str) -> list[tuple[int, int]]:
    """把思考原文切成若干片段区间（纯函数，前后端约定一致）。

    规则：
    1. 先按空行切段；
    2. 若某段内**全部**是编号行且 ≥2 行 → 逐行再拆（对应 ReAct 式 "1. … 2. …" 提纲）；
    3. 丢弃纯空白片段。

    返回 `[(start, end), ...]`：`text` 的字符区间（不含分隔空行），顺序递增、互不重叠。
    """
    segments: list[tuple[int, int]] = []
    for block_start, block_end in _blank_blocks(text):
        lines = _content_lines(text, block_start, block_end)
        if len(lines) >= 2 and all(_ORDERED_LINE.match(text[s:e]) for s, e in lines):
            segments.extend(lines)
        else:
            segments.append((block_start, block_end))
    return segments


def _split_at(segments: list[tuple[int, int]], cuts: set[int]) -> list[tuple[int, int]]:
    """在给定位置切开跨界片段（保证动作能插在段落中间）。"""
    out: list[tuple[int, int]] = []
    for start, end in segments:
        prev = start
        for point in sorted(p for p in cuts if start < p < end):
            out.append((prev, point))
            prev = point
        out.append((prev, end))
    return out


def build_timeline(steps: list[dict] | None, reasoning: str | None) -> list[dict]:
    """把「动作步骤」与「思考片段」合并成一条按位置排序的时间线（纯函数）。

    - `steps`：落库形态的动作列表（retrieval / tool），各带 `reasoning_offset`；缺失按 0 处理；
    - `reasoning`：思考全文；先切片段，再按动作偏移**劈开**跨界片段，最后交错排序；
    - **同一位置时动作在前**：offset=P 表示 `reasoning[0:P]` 在动作之前、`reasoning[P:]` 在其后。

    返回的 dict 键为 snake_case（出参由 Pydantic 转驼峰），只含 `AiStepOut` 的字段。
    """
    text = reasoning or ""

    actions: list[tuple[int, dict]] = []
    for step in steps or []:
        if not step:
            continue
        offset = int(step.get("reasoning_offset") or 0)
        offset = max(0, min(offset, len(text)))
        actions.append(
            (
                offset,
                {
                    "step_id": step.get("step_id") or "step",
                    "kind": step.get("kind") or StepKind.TOOL.value,
                    "status": step.get("status") or "done",
                    "name": step.get("name") or "",
                    "output": step.get("output"),
                    "cost_ms": step.get("cost_ms"),
                    "reasoning_offset": offset,
                },
            )
        )

    segments = _split_at(split_reasoning_segments(text), {pos for pos, _ in actions})

    # (位置, 次序, 载荷)：次序 0=动作、1=思考片段，同位置时动作排前面
    events: list[tuple[int, int, dict]] = [(pos, 0, payload) for pos, payload in actions]
    events += [
        (
            start,
            1,
            {
                "step_id": f"reasoning-{start}",
                "kind": StepKind.THINKING.value,
                "status": "done",
                "start": start,
                "end": end,
            },
        )
        for start, end in segments
    ]
    events.sort(key=lambda item: (item[0], item[1]))
    return [payload for _pos, _order, payload in events]
