"""AI 对话的运行时数据结构与共享常量。

本模块只放**纯数据**（dataclass / 常量），不持有 ORM 对象、不依赖数据库会话，
因此可以安全地被流式生成器跨「请求级会话已归还」的边界使用。

- `LlmRuntime`：`AiChatService._build_llm()` 的产出 —— 客户端 + 用量统计所需的模型/供应商/单价快照；
- `ChatStreamContext`：流式对话预检的产出 —— 流式阶段只依赖它，不再回查数据库；
- `StreamEvent`：SSE 事件包装（`type` 事件名 + `data` 载荷模型）；
- `TokenTimer`：`thinking_ms` / `ttft_ms` 耗时打点（仅流式路径有值）；
- `StepCollector`：工具 / 检索步骤收集器（running→done、单步耗时、推理锚点）。
"""

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import UUID

from langchain_openai import ChatOpenAI

# ---- 共享常量（stream 与 service 两条路径口径一致）----

# 流式结束原因
FINISH_STOP = "stop"
FINISH_ERROR = "error"

# 知识库检索的工具名 / 事件名（预检索与 knowledge_retrieval 工具同名，前端展示统一）
KNOWLEDGE_TOOL = "knowledge_retrieval"


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
        self,
        step_id: str,
        name: str,
        kind: str,
        output: str | None = None,
        sources: list[dict] | None = None,
    ) -> dict:
        """登记完成：返回 done 事件载荷，并把带锚点的记录追加进 `self.steps` 供落库。

        `sources` 是知识库检索的结构化来源；两个 dict **都**带该键
        （`AiStreamToolOut` 已声明 `sources` 字段，因此展开不会报多键）。
        """
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
                "sources": sources,
            }
        )
        return {
            "step_id": step_id,
            "kind": kind,
            "name": name,
            "status": "done",
            "output": output,
            "cost_ms": cost_ms,
            "sources": sources,
        }

    def dump(self) -> list[dict] | None:
        """落库用：空列表返回 None，避免写入无意义的空 JSON。"""
        return self.steps or None


@dataclass
class LlmRuntime:
    """LLM 运行时信息：构造好的客户端 + 用量统计所需的模型维度快照。"""

    llm: ChatOpenAI
    model_code: str
    model_instance_id: UUID | None = None
    provider_id: UUID | None = None
    provider_code: str | None = None
    # 单价快照（元 / 千 token），用于核算成本
    input_price: Decimal | None = None
    output_price: Decimal | None = None


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
    # ---- 用量统计维度快照（写入 ai_usage_record，避免统计时 JOIN / 源数据删除后失真）----
    agent_name: str | None = None
    organization_id: UUID | None = None
    user_id: UUID | None = None
    username: str | None = None
    model_instance_id: UUID | None = None
    provider_id: UUID | None = None
    provider_code: str | None = None
    input_price: Decimal | None = None
    output_price: Decimal | None = None
