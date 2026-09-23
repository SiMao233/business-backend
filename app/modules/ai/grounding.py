"""RAG 证据接地（Grounding）：证据充分性判定 + 引用合法性审计 + 拒答文案 + 出参模型。

定位与 `query.py` 一致 —— **纯逻辑**：不访问数据库、不调用 LLM、不持有会话，
由 chat 层（同步 `chat/service.py` 与流式 `chat/stream.py`）在检索之后调用。
两条对话路径共用同一套判定函数，因此「是否拒答」的口径必然一致
（否则 `done` 事件与同步返回值会给出不同结论）。

在本链路里的位置：

    Query Rewrite（chat 层，只改写检索用 query）
    → retrieve_hits()（Hybrid 召回 + Reranker 精排 + 来源富化）
    → assess_evidence()      ← 本模块：证据是否足以支撑回答
    → should_abstain() ──是─→ 拒答（后端固定文案，**不调用任何 LLM**）
    → grounded prompt（messages.compose_system_prompt / build_messages）
    → LLM
    → audit_answer()         ← 本模块：引用合法性审计（只上报与记日志）

判定规则（刻意保持简单可解释）：
- `hits` 为空 → 证据不足（`no_hits`）；
- 存在有效 `rerank_score` → 只比较**最大值**与 `rag_abstain_min_score`（`ok` / `low_relevance`）；
- 所有 hit 都没有 `rerank_score`（Reranker 未启用）→ **fail-open**（`no_score`，不拒答）：
  向量余弦相似度（实测相关片段仅 0.25~0.6）与关键词命中（无分数）都不可靠，
  拿它们做阈值会大量误杀正常问答；
- 向量分 / 关键词分**不参与**拒答判定。

⚠️ 边界：本模块不修改、不重写、不重新生成正文；也不重新检索、不重新编号 ——
`collect_sources()` 只汇总当前请求**已经产生**的 sources（沿用 `AiSourceOut.index`）。

⚠️ 诚实的定位：提示词约束是**概率性**的，只有 `should_abstain()` 分支给出的
「不猜」是确定性的。本模块的目标是提高答案的 evidence grounding，不是消除幻觉。
"""

import re
from dataclasses import dataclass
from enum import StrEnum

from loguru import logger
from pydantic import Field

from app.common.schema import ApiOutModel
from app.core.config import get_settings

# 知识库检索工具名（必须与 `chat/runtime.KNOWLEDGE_TOOL` 一致）。
# 这里刻意不 import chat 子包：grounding 位于 ai 根目录、被 chat 层依赖，
# 反向依赖会在 chat 层未来引入 grounding 时形成模块环；tests 中有断言守卫两者一致。
KNOWLEDGE_RETRIEVAL_TOOL = "knowledge_retrieval"

# 引用角标：[n]，n 为 1~3 位数字（资料条数不会更多；限位数可避免把 [2024] 之类当引用）
CITATION_RE = re.compile(r"\[(\d{1,3})\]")

# 拒答 / 不确定措辞：含这些措辞的句子不算「无依据的事实性陈述」，不计入 uncited_claims
_REFUSAL_MARKERS = (
    "无法确认",
    "无法回答",
    "无法确定",
    "无法提供",
    "资料不足",
    "没有找到",
    "未检索到",
    "未找到",
)

# 句子切分：句末标点与换行（分号属句内停顿，不切分）
_SENTENCE_SPLIT = re.compile(r"[。！？!?\n]+")

# 计入 uncited_claims 的最短句长：过短的句子（如「好的」「如上」）不视为事实性陈述
UNSUPPORTED_MIN_CHARS = 12

# 知识库检索工具**空命中**时返回给模型的文案（放在本模块而非 chat/messages.py：
# `tools.py` 位于 ai 根目录，若让它 import chat 子包会形成 ai.tools → ai.chat 的反向依赖边）。
# 不能返回空串：模型会把「工具没返回内容」理解为没有约束，转而用预训练知识作答。
NO_EVIDENCE_TOOL_TEXT = (
    "知识库中没有检索到与该问题相关的资料。"
    "请勿基于预训练知识猜测或编造；如果确实没有依据，请直接说明无法确认。"
)


class EvidenceReason(StrEnum):
    """证据充分性判定结论（作为出参 `grounding.reason` 透出）。"""

    OK = "ok"
    NO_HITS = "no_hits"
    LOW_RELEVANCE = "low_relevance"
    NO_SCORE = "no_score"


@dataclass(frozen=True)
class EvidenceAssessment:
    """一次检索的证据充分性判定结果（纯数据）。

    `hit_count` 只统计**后端预检索**的命中条数；模型后续主动调用检索工具追加的来源
    不计入这里，但会计入出参的 `source_count`。
    """

    sufficient: bool
    reason: str
    best_score: float | None = None
    hit_count: int = 0


def best_rerank_score(hits: list[dict] | None) -> float | None:
    """取命中片段里最大的有效 `rerank_score`；无有效分数返回 None（纯函数）。

    `rerank_score` 仅在启用 Reranker 时存在；脏值（bool / 字符串 / 其他非数值）一律忽略，
    避免脏数据把判定带偏。
    """
    scores: list[float] = []
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        score = hit.get("rerank_score")
        if isinstance(score, bool) or not isinstance(score, int | float):
            continue
        scores.append(float(score))
    return max(scores) if scores else None


def assess_evidence(hits: list[dict] | None) -> EvidenceAssessment:
    """判定证据是否足以支撑回答（**不受任何开关影响**，纯函数）。

    开关与工具配置等策略判断放在 `should_abstain()` / `needs_no_evidence_note()`，
    它们内部复用本函数 —— 「是否拒答」与「是否注入无证据说明」两个决策因此同源，不会互相打架。
    """
    items = hits or []
    if not items:
        return EvidenceAssessment(sufficient=False, reason=EvidenceReason.NO_HITS, hit_count=0)

    score = best_rerank_score(items)
    if score is None:
        # Reranker 未启用 / 未返回分数：无法判定相关性 → fail-open，交由提示词约束
        return EvidenceAssessment(
            sufficient=True, reason=EvidenceReason.NO_SCORE, hit_count=len(items)
        )
    if score >= get_settings().rag_abstain_min_score:
        return EvidenceAssessment(
            sufficient=True, reason=EvidenceReason.OK, best_score=score, hit_count=len(items)
        )
    return EvidenceAssessment(
        sufficient=False, reason=EvidenceReason.LOW_RELEVANCE, best_score=score, hit_count=len(items)
    )


def has_non_kb_tools(config: dict | None) -> bool:
    """Agent 是否配置了知识库以外的工具（如 web_search）。"""
    enabled = set((config or {}).get("tools") or [])
    return bool(enabled - {KNOWLEDGE_RETRIEVAL_TOOL})


def should_abstain(hits: list[dict] | None, config: dict | None) -> bool:
    """是否应当直接拒答（后端固定文案，**不调用 LLM**）。

    四类必要条件缺一不可：
    1. 未绑定知识库 → 不介入（普通对话的行为不因本功能改变）；
    2. `rag_grounding_enabled` 与 `rag_abstain_enabled` 均已开启；
    3. 未配置知识库以外的工具（配了 web_search 等工具时先让模型尝试工具，
       改由强约束提示词兜底，否则会误伤多工具 Agent）；
    4. 证据判定为不充分（空命中，或最高 rerank 分低于阈值）。
    """
    if not (config or {}).get("knowledge_ids"):
        return False
    settings = get_settings()
    if not settings.rag_grounding_enabled or not settings.rag_abstain_enabled:
        return False
    if has_non_kb_tools(config):
        return False
    return not assess_evidence(hits).sufficient


def needs_no_evidence_note(hits: list[dict] | None, config: dict | None) -> bool:
    """证据不足但**未**拒答时，提示词里是否需要注入「本轮无证据」的强约束说明。

    覆盖两种降级场景：`rag_abstain_enabled=False`（由模型自行拒答）、
    以及配了非知识库工具（先走工具，但本轮预检索确实没拿到资料）。
    已拒答时不会调用 LLM，也就不需要提示词 → 返回 False。
    """
    if not (config or {}).get("knowledge_ids"):
        return False
    if not get_settings().rag_grounding_enabled:
        return False
    if should_abstain(hits, config):
        return False
    return not assess_evidence(hits).sufficient


def abstain_message() -> str:
    """拒答文案（面向用户，前端直接展示；可用 .env 覆盖）。"""
    return get_settings().rag_abstain_message


def is_refusal(answer: str) -> bool:
    """判断回答是否是「明确拒答 / 声明无法确认」（整段判断，非逐句）。

    供评测框架判「宽松拒答口径」用（`tests/rag/`：进入 ABSTAIN 或正文明确声明无法确认都算通过）。
    与 `audit_answer()` 的拒答措辞白名单**同源**（共用 `_REFUSAL_MARKERS`），
    因此评测口径与线上审计口径不会漂移。
    """
    text = (answer or "").strip()
    if not text:
        return False
    if abstain_message() in text:
        return True
    return any(marker in text for marker in _REFUSAL_MARKERS)


def grounding_prompt_enabled(config: dict | None) -> bool:
    """是否需要注入严格接地提示词（绑定知识库 **且** `rag_grounding_enabled` 开启）。

    供两条对话路径共用：开关判定只在这一处读取，避免两条路径各写一份条件而跑偏。
    """
    if not (config or {}).get("knowledge_ids"):
        return False
    return get_settings().rag_grounding_enabled


@dataclass(frozen=True)
class AnswerAudit:
    """回答的引用审计结果（**只上报，绝不改写正文**）。"""

    cited: list[int]
    invalid_cited: list[int]
    uncited_claims: int
    has_citation: bool


def audit_answer(answer: str, sources: list[dict] | None = None) -> AnswerAudit:
    """审计回答里的引用角标（纯函数）。

    - `cited`：正文出现过的编号（去重升序）；
    - `invalid_cited`：编号存在但**没有对应来源**（即模型编造的引用）；
    - `uncited_claims`：既无 `[n]` 标注、又不含拒答措辞、长度 ≥ `UNSUPPORTED_MIN_CHARS`
      的句子数。**这是启发式观测指标**（只用于统计与告警，不作为拦截依据）：
      它无法判断句子内容是否真的来自资料，只能提示「这句没有标注来源」。
    """
    text = answer or ""
    allowed = {
        int(source["index"])
        for source in (sources or [])
        if isinstance(source, dict) and isinstance(source.get("index"), int)
    }
    cited = sorted({int(match.group(1)) for match in CITATION_RE.finditer(text)})
    invalid = [number for number in cited if number not in allowed]

    uncited = 0
    for raw in _SENTENCE_SPLIT.split(text):
        sentence = raw.strip()
        if len(sentence) < UNSUPPORTED_MIN_CHARS:
            continue
        if CITATION_RE.search(sentence):
            continue
        if any(marker in sentence for marker in _REFUSAL_MARKERS):
            continue
        uncited += 1

    return AnswerAudit(
        cited=cited, invalid_cited=invalid, uncited_claims=uncited, has_citation=bool(cited)
    )


def collect_sources(steps: list[dict] | None) -> list[dict]:
    """汇总本次请求**已经产生**的 sources（不重新检索、不重新编号）。

    `steps` 是 `StepCollector.steps`（落库结构）或同步路径构造的等价 dict 列表；
    非检索类步骤的 `sources` 为 None，直接跳过。`index` 沿用原有值（请求内全局唯一）。
    """
    collected: list[dict] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        sources = step.get("sources")
        if isinstance(sources, list):
            collected.extend(source for source in sources if isinstance(source, dict))
    return collected


# 未做检索判定时（非 RAG 对话：未绑定知识库）使用的占位判定：
# `no_score` 的语义是「无法判定相关性」，与实际不拒答的行为一致。
NO_ASSESSMENT = EvidenceAssessment(sufficient=True, reason=EvidenceReason.NO_SCORE, hit_count=0)


class AiGroundingOut(ApiOutModel):
    """证据接地与引用审计出参（挂在流式 `done` 事件与同步 `AiChatOut` 上）。

    **不落库**（零迁移）：只在实时响应里返回，历史消息靠正文体现（拒答时正文即拒答文案）。
    出现新枚举值 `finish_reason="abstain"`，老客户端按未知值处理即可。
    """

    reason: str = Field(
        default=EvidenceReason.OK.value,
        description="证据判定结论：ok / no_hits / low_relevance / no_score",
    )
    abstained: bool = Field(default=False, description="本次是否因证据不足直接拒答（未调用 LLM）")
    hit_count: int = Field(default=0, description="后端预检索命中条数（不含模型工具追加的来源）")
    source_count: int = Field(
        default=0, description="本次可引用的来源条数（正文 [n] 的合法取值范围）"
    )
    best_score: float | None = Field(
        default=None, description="最高 rerank 相关度；未启用 Reranker 时为 null"
    )
    cited: list[int] = Field(default_factory=list, description="正文实际引用的编号（去重升序）")
    invalid_cited: list[int] = Field(
        default_factory=list, description="正文引用了但无对应来源的编号（审计上报，不修改正文）"
    )
    uncited_claims: int = Field(
        default=0, description="无 [n] 标注的事实性句子数（启发式观测指标，非拦截依据）"
    )


def grounding_out(
    assessment: EvidenceAssessment | None,
    sources: list[dict] | None,
    answer: str,
    *,
    abstained: bool = False,
) -> AiGroundingOut | None:
    """组装 grounding 出参（`rag_grounding_enabled=False` 时返回 None，保持改造前行为）。

    `assessment` 传 None 表示本次未做检索判定（未绑定知识库的非 RAG 对话），
    按 `NO_ASSESSMENT` 处理 —— 只上报审计信息，声称拒答结论。
    唯一的副作用：发现编造引用时记一条 warning —— 放在这里而非调用方，
    是为了让两条对话路径共用同一处打点，避免一边打了另一边漏掉。
    """
    if not get_settings().rag_grounding_enabled:
        return None
    collected = sources or []
    audit = audit_answer(answer, collected)
    if audit.invalid_cited:
        logger.warning(
            "回答引用了不存在的来源 indices={} 可用来源条数={}",
            audit.invalid_cited,
            len(collected),
        )
    resolved = assessment or NO_ASSESSMENT
    return AiGroundingOut(
        reason=resolved.reason,
        abstained=abstained,
        hit_count=resolved.hit_count,
        source_count=len(collected),
        best_score=resolved.best_score,
        cited=audit.cited,
        invalid_cited=audit.invalid_cited,
        uncited_claims=audit.uncited_claims,
    )
