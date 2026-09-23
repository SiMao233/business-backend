"""RAG 评测内核：Golden Dataset 模型 + 检索/答案指标 + 报告与基线对比（纯逻辑）。

定位：**纯函数模块** —— 不访问数据库、不调用 LLM、不被 chat 链路引用，
因此引入评测框架对线上行为零影响（唯一的「副作用」是 `config_snapshot()` 读取配置）。

三层职责：
1. **数据集**：`GoldenQuestion` / `GoldenDataset` / `GoldenFile` + `load_golden_file()`（TOML + 严格校验）；
2. **指标**：命中判定 / Recall@K / MRR（检索层）、correctness / citation（答案层），全部纯函数；
3. **报告**：`EvalReport`（可落 JSON）+ `compare_reports()` + `render_*()`（逐指标 delta 与逐题改善/回归）。

⚠️ 关键设计：**ground truth 用「证据子串 + 文档名」，不用 chunk_id**
重新切分（Chunking 改动）会让 chunk 主键全部改变 —— 硬编码 chunk_id 的金标会一次性失效，
而「评估 Chunking 改动」正是本框架的首要目标。故命中判定按**内容**（`expected_evidence`
必须出现在命中片段里）与**文档名**（`expected_source`）进行，与 ID 解耦。

⚠️ 指标是**确定性 proxy**，不是语义级判断：答案层用「金标要点命中率 + 引用合法性 + 拒答口径」
近似 correctness / faithfulness / citation，真假阳性仍需人工复核表兜底
（用法与局限性见 `tests/rag/README.md`）。

⚠️ 数据集格式用 TOML（标准库 `tomllib` 解析）而非 YAML：**零新增依赖**（项目未装 pyyaml）。
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.modules.ai.grounding import audit_answer, is_refusal

# 默认 Recall@K 的 K 列表（评测时可一次性算出多个 K，避免重复检索）
DEFAULT_KS = (1, 3, 5, 10)

# 默认数据集路径（仓库根 / tests/rag/golden/rag_golden.toml）
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[3] / "tests" / "rag" / "golden" / "rag_golden.toml"

# 检索指标里「越低越好」的例外项（其余指标一律越高越好）
LOWER_IS_BETTER = frozenset({"uncited_claims_total", "false_abstain_rate"})


class DatasetError(ValueError):
    """Golden Dataset 校验失败（携带可直接排查的字段级原因）。"""


class Category(StrEnum):
    """题目分类（与评测报告的分组口径一致）。"""

    SIMPLE_FACT = "simple_fact"
    KEYWORD = "keyword"
    PROPER_NOUN = "proper_noun"
    MULTI_TURN = "multi_turn"
    ANAPHORA = "anaphora"
    NO_ANSWER = "no_answer"
    CROSS_CHUNK = "cross_chunk"


class ExpectedBehavior(StrEnum):
    """期望行为：作答 / 拒答（无答案题）。"""

    ANSWER = "answer"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class GoldenQuestion:
    """一道金标题（字段语义见 `tests/rag/README.md`）。

    `expected_evidence` 是**必须出现在命中片段里的原文档特征句**（可多条，跨 chunk 题就靠它测「漏了几条」）；
    `expected_source` 是来源文档名（比 document_id 稳定）；两者都与 chunk 主键无关，重切分后依然有效。
    """

    id: str
    category: str
    question: str
    expected_answer: str = ""
    expected_source: tuple[str, ...] = ()
    expected_evidence: tuple[str, ...] = ()
    expected_answer_points: tuple[str, ...] = ()
    expected_behavior: str = ExpectedBehavior.ANSWER.value
    history: tuple[tuple[str, str], ...] = ()
    notes: str = ""

    @property
    def is_no_answer(self) -> bool:
        """无答案题：期望拒答，不参与检索指标（没有 evidence 可算 Recall）。"""
        return self.expected_behavior == ExpectedBehavior.ABSTAIN.value

    def answer_points(self) -> tuple[str, ...]:
        """correctness 用的金标要点：未单独给 points 时回退为整段 expected_answer。"""
        if self.expected_answer_points:
            return self.expected_answer_points
        return (self.expected_answer,) if self.expected_answer else ()


@dataclass(frozen=True)
class GoldenDataset:
    """一个知识库对应的一组金标题。"""

    name: str
    knowledge_base_id: str
    agent_id: str = ""
    questions: tuple[GoldenQuestion, ...] = ()


@dataclass(frozen=True)
class GoldenFile:
    """Golden Dataset 文件内容（版本 + 阈值 + 多个数据集）。"""

    version: int
    thresholds: dict[str, float] = field(default_factory=dict)
    datasets: tuple[GoldenDataset, ...] = ()

    @property
    def questions(self) -> tuple[GoldenQuestion, ...]:
        return tuple(q for dataset in self.datasets for q in dataset.questions)

    def categories(self) -> dict[str, int]:
        """各分类题数（便于观察数据集分布是否失衡）。"""
        counts: dict[str, int] = {}
        for question in self.questions:
            counts[question.category] = counts.get(question.category, 0) + 1
        return counts


# ---- 数据集加载与校验 ----


def load_golden_file(path: str | Path = DEFAULT_DATASET_PATH) -> GoldenFile:
    """读取并**严格校验** Golden Dataset（任一问题都抛 `DatasetError`，消息含字段路径）。

    刻意不做「容错解析」：数据集是人工维护的金标准，字段拼错/漏填必须立刻失败，
    否则会静默地把题目变成永不命中的噪音。
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise DatasetError(f"数据集文件不存在: {dataset_path}")
    try:
        raw = tomllib.loads(dataset_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:  # pragma: no cover - 由 tomllib 保证
        raise DatasetError(f"TOML 解析失败: {dataset_path}: {exc}") from exc
    return parse_golden(raw, source=str(dataset_path))


def parse_golden(raw: Mapping[str, Any], *, source: str = "<memory>") -> GoldenFile:
    """把 TOML 反序列化后的 dict 校验成 `GoldenFile`（纯函数，便于单测直接用内存数据）。"""
    problems: list[str] = []

    version = raw.get("version")
    if not isinstance(version, int):
        problems.append("`version` 必须是整数")

    unknown_top = set(raw) - {"version", "thresholds", "datasets"}
    if unknown_top:
        problems.append(f"未知顶层键（疑似拼写错误）: {sorted(unknown_top)}")

    thresholds = _parse_thresholds(raw.get("thresholds"), problems)

    datasets: list[GoldenDataset] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(_as_list(raw.get("datasets"))):
        where = f"datasets[{index}]"
        if not isinstance(item, Mapping):
            problems.append(f"{where} 必须是表（table）")
            continue
        name = _require_str(item.get("name"), f"{where}.name", problems)
        kb_id = _require_str(item.get("knowledge_base_id"), f"{where}.knowledge_base_id", problems)
        unknown = set(item) - {"name", "knowledge_base_id", "agent_id", "questions"}
        if unknown:
            problems.append(f"{where} 含未知键: {sorted(unknown)}")

        questions: list[GoldenQuestion] = []
        for q_index, q_item in enumerate(_as_list(item.get("questions"))):
            q_where = f"{where}.questions[{q_index}]"
            question = _parse_question(q_item, q_where, problems, seen_ids)
            if question is not None:
                questions.append(question)
        datasets.append(
            GoldenDataset(
                name=name,
                knowledge_base_id=kb_id,
                agent_id=str(item.get("agent_id") or ""),
                questions=tuple(questions),
            )
        )

    if not datasets:
        problems.append("`datasets` 至少需要一项")
    if problems:
        detail = "\n".join(f"  - {problem}" for problem in problems)
        raise DatasetError(f"Golden Dataset 校验失败（{source}）:\n{detail}")

    return GoldenFile(
        version=int(version), thresholds=thresholds, datasets=tuple(datasets)
    )


def _parse_thresholds(raw: Any, problems: list[str]) -> dict[str, float]:
    """阈值为「指标名 → 下限」的映射，仅接受数值。"""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        problems.append("`thresholds` 必须是表（table）")
        return {}
    values: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            problems.append(f"thresholds.{key} 必须是数值")
            continue
        values[str(key)] = float(value)
    return values


def _parse_question(
    raw: Any, where: str, problems: list[str], seen_ids: set[str]
) -> GoldenQuestion | None:
    """校验单题；返回 None 表示该题有问题（问题已记入 `problems`）。"""
    if not isinstance(raw, Mapping):
        problems.append(f"{where} 必须是表（table）")
        return None

    unknown = set(raw) - {
        "id",
        "category",
        "question",
        "expected_answer",
        "expected_source",
        "expected_evidence",
        "expected_answer_points",
        "expected_behavior",
        "history",
        "notes",
    }
    if unknown:
        problems.append(f"{where} 含未知键: {sorted(unknown)}")

    question_id = _require_str(raw.get("id"), f"{where}.id", problems)
    if question_id in seen_ids:
        problems.append(f"{where}.id 重复: {question_id}")
    seen_ids.add(question_id)

    category = _require_str(raw.get("category"), f"{where}.category", problems)
    if category and category not in set(Category):
        problems.append(f"{where}.category 非法: {category}（可选: {[c.value for c in Category]}）")

    text = _require_str(raw.get("question"), f"{where}.question", problems)
    behavior = str(raw.get("expected_behavior") or ExpectedBehavior.ANSWER.value)
    if behavior not in set(ExpectedBehavior):
        problems.append(f"{where}.expected_behavior 非法: {behavior}")

    evidence = _str_tuple(raw.get("expected_evidence"), f"{where}.expected_evidence", problems)
    sources = _str_tuple(raw.get("expected_source"), f"{where}.expected_source", problems)
    points = _str_tuple(raw.get("expected_answer_points"), f"{where}.expected_answer_points", problems)
    expected_answer = str(raw.get("expected_answer") or "")
    history = _parse_history(raw.get("history"), where, problems)

    is_no_answer = behavior == ExpectedBehavior.ABSTAIN.value
    if is_no_answer:
        if evidence:
            problems.append(f"{where}: 无答案题不应填 expected_evidence（期望检索不到证据）")
        if sources:
            problems.append(f"{where}: 无答案题不应填 expected_source")
    else:
        if not evidence:
            problems.append(f"{where}.expected_evidence 不能为空（非无答案题必须给出证据子串）")
        if not sources:
            problems.append(f"{where}.expected_source 不能为空（非无答案题必须给出期望文档名）")
        if not expected_answer and not points:
            problems.append(f"{where}: expected_answer 与 expected_answer_points 至少要有一个")

    if category in (Category.MULTI_TURN.value, Category.ANAPHORA.value) and not history:
        problems.append(f"{where}: 分类 {category} 必须提供 history（多轮/指代题的前提）")

    return GoldenQuestion(
        id=question_id,
        category=category,
        question=text,
        expected_answer=expected_answer,
        expected_source=sources,
        expected_evidence=evidence,
        expected_answer_points=points,
        expected_behavior=behavior,
        history=history,
        notes=str(raw.get("notes") or ""),
    )


def _parse_history(raw: Any, where: str, problems: list[str]) -> tuple[tuple[str, str], ...]:
    """历史消息：[["user", "..."], ["assistant", "..."]]。"""
    if raw is None:
        return ()
    turns: list[tuple[str, str]] = []
    for index, item in enumerate(_as_list(raw)):
        if not isinstance(item, list | tuple) or len(item) != 2:
            problems.append(f"{where}.history[{index}] 必须是 [role, content] 二元组")
            continue
        role, content = str(item[0]), str(item[1])
        if role not in {"user", "assistant"}:
            problems.append(f"{where}.history[{index}] role 非法: {role}")
        turns.append((role, content))
    return tuple(turns)


def _as_list(raw: Any) -> list[Any]:
    if raw is None:
        return []
    return list(raw) if isinstance(raw, list | tuple) else []


def _require_str(raw: Any, where: str, problems: list[str]) -> str:
    if not isinstance(raw, str) or not raw.strip():
        problems.append(f"{where} 不能为空")
        return ""
    return raw.strip()


def _str_tuple(raw: Any, where: str, problems: list[str]) -> tuple[str, ...]:
    """字符串数组：允许标量（单条）与数组两种写法，元素必须是非空字符串。"""
    if raw is None:
        return ()
    items = raw if isinstance(raw, list | tuple) else [raw]
    values: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            problems.append(f"{where} 的元素必须是非空字符串")
            continue
        values.append(item.strip())
    return tuple(values)


# ---- 检索层指标（纯函数）----


def _content_of(hit: Any) -> str:
    """取命中片段的正文（hit 为 retrieve_hits 产出的 dict）。"""
    if not isinstance(hit, Mapping):
        return ""
    return str(hit.get("content") or "")


def evidence_matches(content: str, expected_evidence: Sequence[str]) -> tuple[str, ...]:
    """该段正文命中了哪些证据条目（按 `expected_evidence` 顺序返回）。"""
    text = content or ""
    return tuple(item for item in expected_evidence if item and item in text)


def first_evidence_rank(hits: Sequence[Any], expected_evidence: Sequence[str]) -> int | None:
    """第一个命中**任一**证据的片段排名（1-based）；无命中返回 None。"""
    for rank, hit in enumerate(hits, start=1):
        if evidence_matches(_content_of(hit), expected_evidence):
            return rank
    return None


def hit_at_k(hits: Sequence[Any], expected_evidence: Sequence[str], k: int) -> float:
    """top-K 内是否命中任一证据（1.0/0.0）。"""
    rank = first_evidence_rank(hits[:k], expected_evidence)
    return 1.0 if rank is not None else 0.0


def evidence_recall_at_k(hits: Sequence[Any], expected_evidence: Sequence[str], k: int) -> float:
    """top-K 覆盖的证据条目比例（跨 chunk 题因此可测「漏了几条」）。

    无证据条目时返回 1.0（真空真值）；调用方应改用 `retrieval_scores()` 自动跳过这类题。
    """
    if not expected_evidence:
        return 1.0
    found = {item for hit in hits[:k] for item in evidence_matches(_content_of(hit), expected_evidence)}
    return len(found) / len(expected_evidence)


def source_recall_at_k(hits: Sequence[Any], expected_source: Sequence[str], k: int) -> float:
    """top-K 覆盖的期望文档比例（比「文档命中 0/1」更能反映跨文档题的遗漏）。"""
    if not expected_source:
        return 1.0
    names = {
        str(hit.get("document_name") or "")
        for hit in hits[:k]
        if isinstance(hit, Mapping)
    }
    return len(names & set(expected_source)) / len(expected_source)


def reciprocal_rank(hits: Sequence[Any], expected_evidence: Sequence[str]) -> float:
    """首个命中证据的排名倒数（无命中 = 0.0）。"""
    rank = first_evidence_rank(hits, expected_evidence)
    return 1.0 / rank if rank else 0.0


def retrieval_scores(
    hits: Sequence[Any], question: GoldenQuestion, ks: Sequence[int] = DEFAULT_KS
) -> dict[str, float] | None:
    """单题检索指标；**无 `expected_evidence` 的题（无答案题）返回 None**（不参与检索聚合）。"""
    evidence = question.expected_evidence
    if not evidence:
        return None
    scores: dict[str, float] = {}
    for k in ks:
        scores[f"hit@{k}"] = hit_at_k(hits, evidence, k)
        scores[f"evidence_recall@{k}"] = evidence_recall_at_k(hits, evidence, k)
        scores[f"source_recall@{k}"] = source_recall_at_k(hits, question.expected_source, k)
    scores["mrr"] = reciprocal_rank(hits, evidence)
    return scores


# ---- 答案层指标（纯函数，确定性 proxy）----


@dataclass(frozen=True)
class CitationMetrics:
    """引用审计结果（由 `grounding.audit_answer` 产出，保证与线上口径一致）。"""

    cited: tuple[int, ...] = ()
    invalid_cited: tuple[int, ...] = ()
    uncited_claims: int = 0
    has_citation: bool = False

    @property
    def valid(self) -> bool:
        """引用是否全部有对应来源（编造引用 = 不合格）。"""
        return not self.invalid_cited


def citation_metrics(answer: str, sources: Sequence[Mapping[str, Any]] | None = None) -> CitationMetrics:
    """引用审计（复用线上 `audit_answer`，只审计不改写）。"""
    audit = audit_answer(answer or "", [dict(source) for source in (sources or [])])
    return CitationMetrics(
        cited=tuple(audit.cited),
        invalid_cited=tuple(audit.invalid_cited),
        uncited_claims=audit.uncited_claims,
        has_citation=audit.has_citation,
    )


def answer_correctness(answer: str, expected_points: Sequence[str]) -> float | None:
    """金标要点命中率（无要点可判时返回 None）。"""
    points = [point for point in expected_points if point]
    if not points:
        return None
    text = answer or ""
    return sum(1 for point in points if point in text) / len(points)


def points_supported_by_context(
    answer: str, expected_points: Sequence[str], context: str
) -> float | None:
    """答案命中的要点里，有多少**同时出现在本轮 context**（低 = 疑似幻觉，仅观测不设门禁）。"""
    hit_points = [point for point in expected_points if point and point in (answer or "")]
    if not hit_points:
        return None
    ctx = context or ""
    return sum(1 for point in hit_points if point in ctx) / len(hit_points)


def abstain_pass(answer: str, *, abstained: bool) -> bool:
    """无答案题的**宽松**通过口径：进入 ABSTAIN 或正文明确声明无法确认，均算通过。"""
    return abstained or is_refusal(answer)


# ---- 聚合与报告 ----


def macro_average(rows: Iterable[Mapping[str, float | None]]) -> dict[str, float]:
    """按指标求均值（macro：先每题算分再平均；`None` 跳过，全为 None 的指标不出现）。"""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for name, value in row.items():
            if value is None:
                continue
            totals[name] = totals.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0) + 1
    return {name: totals[name] / counts[name] for name in totals}


# 阈值键前缀 → 报告指标名前缀（数据集里用 snake_case 便于手写，报告里用 `@` 便于阅读）
_THRESHOLD_PREFIXES = ("hit_at_", "evidence_recall_at_", "source_recall_at_")


def threshold_to_metric(key: str) -> str:
    """数据集阈值键（snake_case）→ 报告指标名：`hit_at_5` → `hit@5`。

    两处命名不一致是刻意的（手写友好 vs 报告可读），但**必须有显式映射**：
    直接拿 `hit_at_5` 去报告里取值会得到 None，门禁会静默失效或永远报错。
    无法识别的键原样返回（由 `known_threshold_keys()` 校验拦下）。
    """
    for prefix in _THRESHOLD_PREFIXES:
        if key.startswith(prefix):
            return f"{prefix[:-4]}@{key[len(prefix) :]}"
    return key


def known_threshold_keys(ks: Sequence[int] = DEFAULT_KS) -> set[str]:
    """数据集里允许出现的全部阈值键（拼错会被 `test_golden_dataset.py` 拦下）。"""
    keys = {"mrr"}
    for k in ks:
        keys |= {f"hit_at_{k}", f"evidence_recall_at_{k}", f"source_recall_at_{k}"}
    return keys


def group_by_category(
    per_question: Sequence[Mapping[str, Any]], metric_prefix: str = "retrieval"
) -> dict[str, dict[str, float]]:
    """按分类汇总（每个分类一组 macro 均值，另附 `_count` = 该分类参评题数）。"""
    buckets: dict[str, list[Mapping[str, float | None]]] = {}
    for record in per_question:
        scores = record.get(metric_prefix)
        if not isinstance(scores, Mapping):
            continue
        buckets.setdefault(str(record.get("category") or "unknown"), []).append(scores)
    grouped: dict[str, dict[str, float]] = {}
    for category, rows in sorted(buckets.items()):
        grouped[category] = {**macro_average(rows), "_count": float(len(rows))}
    return grouped


@dataclass
class EvalReport:
    """一次评测的完整结果（可 `to_dict()` 落 JSON，也可 `from_dict()` 读回做基线对比）。"""

    meta: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    answers: dict[str, Any] | None = None
    per_question: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "retrieval": self.retrieval,
            "answers": self.answers,
            "per_question": self.per_question,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalReport:
        return cls(
            meta=dict(data.get("meta") or {}),
            retrieval=dict(data.get("retrieval") or {}),
            answers=dict(data["answers"]) if data.get("answers") else None,
            per_question=[dict(item) for item in (data.get("per_question") or [])],
        )


def config_snapshot(keys: Sequence[str] = ()) -> dict[str, Any]:
    """记录影响检索/生成的配置快照（两次运行的 delta 只有配置相同才可比）。

    `keys` 为空时取一组默认关注项（RAG 检索、重排、改写、切分、默认模型）。
    """
    from app.core.config import get_settings

    settings = get_settings()
    targets = list(keys) or [
        "rag_top_k",
        "rag_score_threshold",
        "rag_hybrid_enabled",
        "rag_candidate_k",
        "rag_rerank_enabled",
        "rag_rerank_score_threshold",
        "rag_abstain_enabled",
        "rag_abstain_min_score",
        "rag_grounding_enabled",
        "rag_query_rewrite_enabled",
        "chunk_size",
        "chunk_overlap",
        "embedding_batch_size",
    ]
    return {key: getattr(settings, key, None) for key in targets}


# ---- 基线对比 ----


@dataclass(frozen=True)
class MetricDelta:
    """单个指标的基线对比结果。"""

    name: str
    baseline: float | None
    current: float | None
    delta: float | None

    @property
    def direction(self) -> str:
        """improved / regressed / unchanged / added / removed。"""
        if self.baseline is None:
            return "added"
        if self.current is None:
            return "removed"
        if self.delta is None or abs(self.delta) < 1e-9:
            return "unchanged"
        higher_is_better = self.name not in LOWER_IS_BETTER
        improved = (self.delta > 0) if higher_is_better else (self.delta < 0)
        return "improved" if improved else "regressed"


@dataclass(frozen=True)
class QuestionChange:
    """单题层面的变化（用于定位「哪几道题变好/变差」）。"""

    question_id: str
    category: str
    metric: str
    baseline: float | None
    current: float | None


@dataclass(frozen=True)
class ReportDiff:
    """两份报告的差异（指标层 + 题目层）。"""

    metric_deltas: tuple[MetricDelta, ...] = ()
    improved_questions: tuple[QuestionChange, ...] = ()
    regressed_questions: tuple[QuestionChange, ...] = ()
    added_questions: tuple[str, ...] = ()
    removed_questions: tuple[str, ...] = ()
    baseline_meta: dict[str, Any] = field(default_factory=dict)
    current_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def has_regression(self) -> bool:
        """是否存在指标回退（供 `--fail-on-regression` 决定退出码）。"""
        return any(delta.direction == "regressed" for delta in self.metric_deltas)


def compare_metrics(
    baseline: Mapping[str, float | None], current: Mapping[str, float | None]
) -> tuple[MetricDelta, ...]:
    """逐指标对比（并集，保持基线顺序、新增指标排后面）。"""
    names = list(baseline) + [name for name in current if name not in baseline]
    deltas: list[MetricDelta] = []
    for name in names:
        before = baseline.get(name)
        after = current.get(name)
        delta = None if before is None or after is None else float(after) - float(before)
        deltas.append(
            MetricDelta(
                name=name,
                baseline=None if before is None else float(before),
                current=None if after is None else float(after),
                delta=delta,
            )
        )
    return tuple(deltas)


def compare_reports(
    baseline: EvalReport, current: EvalReport, *, question_metric: str = "hit@5"
) -> ReportDiff:
    """对比两份报告：整体指标 delta + 逐题 hit/miss 变化（哪个指标由 `question_metric` 指定）。"""
    overall_baseline = dict((baseline.retrieval or {}).get("overall") or {})
    overall_current = dict((current.retrieval or {}).get("overall") or {})
    if baseline.answers or current.answers:
        for name, value in ((baseline.answers or {}).get("overall") or {}).items():
            overall_baseline.setdefault(f"answers.{name}", value)
        for name, value in ((current.answers or {}).get("overall") or {}).items():
            overall_current.setdefault(f"answers.{name}", value)

    before = _question_metric_map(baseline.per_question, question_metric)
    after = _question_metric_map(current.per_question, question_metric)
    improved: list[QuestionChange] = []
    regressed: list[QuestionChange] = []
    for question_id in sorted(set(before) & set(after)):
        old, new = before[question_id], after[question_id]
        if old["value"] == new["value"]:
            continue
        change = QuestionChange(
            question_id=question_id,
            category=str(new.get("category") or ""),
            metric=question_metric,
            baseline=old["value"],
            current=new["value"],
        )
        (improved if new["value"] > old["value"] else regressed).append(change)

    return ReportDiff(
        metric_deltas=compare_metrics(overall_baseline, overall_current),
        improved_questions=tuple(improved),
        regressed_questions=tuple(regressed),
        added_questions=tuple(sorted(set(after) - set(before))),
        removed_questions=tuple(sorted(set(before) - set(after))),
        baseline_meta=dict(baseline.meta),
        current_meta=dict(current.meta),
    )


def _question_metric_map(
    per_question: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, dict[str, Any]]:
    """{题目 id → {category, value}}；缺指标或无检索结果的题**跳过**（避免逐题对比报 KeyError）。"""
    out: dict[str, dict[str, Any]] = {}
    for record in per_question:
        question_id = str(record.get("id") or "")
        scores = record.get("retrieval")
        if not question_id or not isinstance(scores, Mapping):
            continue
        value = scores.get(metric)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        out[question_id] = {"category": str(record.get("category") or ""), "value": float(value)}
    return out


# ---- 渲染（纯字符串，无打印副作用）----


def render_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], *, indent: str = "  ") -> str:
    """极简等宽表格（不引第三方依赖）；行长度不足时按空串补齐。"""
    width = len(headers)
    cells = [[str(cell) for cell in row][:width] + [""] * max(0, width - len(row)) for row in rows]
    widths = [len(str(headers[i])) for i in range(width)]
    for row in cells:
        for index in range(width):
            widths[index] = max(widths[index], len(row[index]))
    lines = [
        indent + "  ".join(str(headers[i]).ljust(widths[i]) for i in range(width)),
        indent + "  ".join("-" * widths[i] for i in range(width)),
    ]
    lines.extend(indent + "  ".join(row[i].ljust(widths[i]) for i in range(width)) for row in cells)
    return "\n".join(lines)


def render_summary(report: EvalReport, *, ks: Sequence[int] = DEFAULT_KS) -> str:
    """单次运行的指标摘要（整体 + 分类 + 无答案题观测 + 原始问题对照）。"""
    lines: list[str] = ["检索层指标（macro）"]
    overall = (report.retrieval or {}).get("overall") or {}
    keys = [f"hit@{k}" for k in ks] + [f"evidence_recall@{k}" for k in ks] + ["mrr"]
    lines.append(render_table(["指标", "值"], [[key, _fmt(overall.get(key))] for key in keys if key in overall]))

    by_category = (report.retrieval or {}).get("by_category") or {}
    if by_category:
        lines.append("按分类（hit@5 / evidence_recall@5 / mrr / 题数）")
        lines.append(
            render_table(
                ["分类", "hit@5", "evid@5", "mrr", "题数"],
                [
                    [
                        category,
                        _fmt(scores.get("hit@5")),
                        _fmt(scores.get("evidence_recall@5")),
                        _fmt(scores.get("mrr")),
                        str(int(scores.get("_count", 0))),
                    ]
                    for category, scores in by_category.items()
                ],
            )
        )

    no_answer = (report.retrieval or {}).get("no_answer") or {}
    if no_answer:
        lines.append("无答案题（不参与 Recall，仅观测：理想情况是零命中率高）")
        lines.append(
            render_table(
                ["题数", "零命中率", "低相关率"],
                [
                    [
                        str(int(no_answer.get("count", 0))),
                        _fmt(no_answer.get("no_hits_rate")),
                        _fmt(no_answer.get("low_relevance_rate")),
                    ]
                ],
            )
        )

    raw_overall = (report.retrieval or {}).get("raw_overall") or {}
    if raw_overall and any(raw_overall.get(key) != overall.get(key) for key in keys):
        lines.append("原始问题口径对照（改写前 vs 线上；仅对有历史的题可能有差异）")
        lines.append(
            render_table(
                ["指标", "原始问题", "线上"],
                [[key, _fmt(raw_overall.get(key)), _fmt(overall.get(key))] for key in keys if key in overall],
            )
        )

    if report.answers:
        lines.append("答案层指标（宏观）")
        lines.append(
            render_table(
                ["指标", "值"],
                [[name, _fmt(value)] for name, value in (report.answers.get("overall") or {}).items()],
            )
        )
    return "\n".join(lines)


def render_diff(diff: ReportDiff) -> str:
    """基线对比的结果（指标 delta + 逐题改善/回归清单）。"""
    lines = ["指标对比（baseline → current）"]
    lines.append(
        render_table(
            ["指标", "baseline", "current", "delta", ""],
            [
                [
                    delta.name,
                    _fmt(delta.baseline),
                    _fmt(delta.current),
                    _fmt(delta.delta, signed=True),
                    _arrow(delta.direction),
                ]
                for delta in diff.metric_deltas
            ],
        )
    )
    if diff.regressed_questions:
        lines.append(f"回归题目（{len(diff.regressed_questions)}）")
        lines.append(
            render_table(
                ["题目", "分类", "指标", "baseline", "current"],
                [
                    [c.question_id, c.category, c.metric, _fmt(c.baseline), _fmt(c.current)]
                    for c in diff.regressed_questions[:20]
                ],
            )
        )
    if diff.improved_questions:
        lines.append(f"改善题目（{len(diff.improved_questions)}）")
        lines.append(
            render_table(
                ["题目", "分类", "指标", "baseline", "current"],
                [
                    [c.question_id, c.category, c.metric, _fmt(c.baseline), _fmt(c.current)]
                    for c in diff.improved_questions[:20]
                ],
            )
        )
    if diff.added_questions or diff.removed_questions:
        lines.append(
            f"新增题目 {len(diff.added_questions)} / 移除题目 {len(diff.removed_questions)}"
        )
    return "\n".join(lines)


def _fmt(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):  # pragma: no cover - 报告里只会有数值
        return str(value)
    return f"{number:+.3f}" if signed else f"{number:.3f}"


def _arrow(direction: str) -> str:
    return {"improved": "▲", "regressed": "▼", "unchanged": "=", "added": "new", "removed": "gone"}.get(
        direction, ""
    )


# ---- 人工复核表（生成 → 人工填写 → 回读汇总）----


# 人工结论列允许的取值（其它值视为未填，避免自由文本污染统计）
REVIEW_PASS = "通过"
REVIEW_FAIL = "不通过"
REVIEW_UNSURE = "存疑"
REVIEW_VERDICTS = (REVIEW_PASS, REVIEW_FAIL, REVIEW_UNSURE)

# 复核表列名（生成与解析共用，改名会同时影响两边）
REVIEW_ID_COLUMN = "题号"
REVIEW_VERDICT_COLUMN = "人工结论"

# 复核表里「实际答案」列的截断长度（表格要能看，不能塞进整段回答）
REVIEW_ANSWER_LIMIT = 300


def _clip(value: Any, limit: int) -> str:
    """按长度截断（供复核表控制列宽）；转义交给 `render_markdown_table`，勿重复处理。"""
    text = str(value if value is not None else "")
    return text[:limit] + "…" if limit and len(text) > limit else text


def _cell(value: Any) -> str:
    """单元格转义：换行压成空格、`|` 转义，避免破坏 Markdown 表格结构。"""
    return str(value if value is not None else "").replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def render_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """生成 Markdown 表格（单元格已转义）。"""
    lines = [
        "| " + " | ".join(_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [_cell(cell) for cell in row]
        cells += [""] * (len(headers) - len(cells))
        lines.append("| " + " | ".join(cells[: len(headers)]) + " |")
    return "\n".join(lines)


def render_review_markdown(report: EvalReport) -> str:
    """生成人工复核表：每题一行，含自动指标与**留空的「人工结论」列**。

    为什么必须有人工一列：自动指标都是**确定性 proxy**（要点命中率、引用合法性、拒答措辞），
    它们能拦住明显退化，但判断不了「答得对不对、全不全」。人工填完 `通过` / `不通过` / `存疑`
    后用 `parse_review_markdown()` + `summarize_review()` 汇总。
    """
    rows: list[list[Any]] = []
    for record in report.per_question:
        retrieval = record.get("retrieval") or {}
        answer = record.get("answer") or {}
        citations = answer.get("citations") or {}
        rows.append(
            [
                record.get("id", ""),
                record.get("category", ""),
                _clip(record.get("question", ""), 120),
                _clip(record.get("expected_answer", ""), 120),
                _clip(answer.get("answer", ""), REVIEW_ANSWER_LIMIT),
                _fmt(retrieval.get("hit@5")) if retrieval else "-",
                _fmt(answer.get("correctness")) if answer else "-",
                "是" if citations.get("has_citation") else "否",
                "" if not citations else ("-" if not citations.get("invalid_cited") else "有编造"),
                "",  # 人工结论（留空待填）
                "",  # 备注
            ]
        )
    headers = [
        REVIEW_ID_COLUMN,
        "分类",
        "问题",
        "期望答案",
        "实际答案",
        "hit@5",
        "要点命中",
        "有引用",
        "引用审计",
        REVIEW_VERDICT_COLUMN,
        "备注",
    ]
    title = "# RAG 人工复核表\n\n"
    hint = (
        f"> 填写 `{REVIEW_VERDICT_COLUMN}` 列（`{REVIEW_PASS}` / `{REVIEW_FAIL}` / "
        f"`{REVIEW_UNSURE}`），然后用 `--review-load <本文件>` 汇总通过率。\n"
        f"> 自动列仅供参考：`hit@5` / `要点命中` 是确定性 proxy。\n\n"
    )
    return title + hint + render_markdown_table(headers, rows) + "\n"


def parse_review_markdown(text: str) -> dict[str, str]:
    """回读人工复核表 → `{题目 id: 人工结论}`（只收 `通过`/`不通过`/`存疑`，其余视为未填）。"""
    header: list[str] | None = None
    verdicts: dict[str, str] = {}
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if all(set(cell) <= {"-", ":"} and cell for cell in cells):
            continue  # 分隔行
        if REVIEW_ID_COLUMN not in header or REVIEW_VERDICT_COLUMN not in header:
            continue
        question_id = cells[header.index(REVIEW_ID_COLUMN)].replace("\\|", "|")
        verdict = cells[header.index(REVIEW_VERDICT_COLUMN)].replace("\\|", "|")
        if question_id and verdict in REVIEW_VERDICTS:
            verdicts[question_id] = verdict
    return verdicts


def summarize_review(verdicts: Mapping[str, str]) -> dict[str, Any]:
    """汇总人工复核：各结论计数 + 通过率（**未填的题不计入分母**）。"""
    counts = dict.fromkeys(REVIEW_VERDICTS, 0)
    for verdict in verdicts.values():
        if verdict in counts:
            counts[verdict] += 1
    filled = sum(counts.values())
    return {
        "filled": filled,
        "pass": counts[REVIEW_PASS],
        "fail": counts[REVIEW_FAIL],
        "unsure": counts[REVIEW_UNSURE],
        "pass_rate": (counts[REVIEW_PASS] / filled) if filled else None,
    }
