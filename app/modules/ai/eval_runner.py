"""RAG 评测执行层：把「数据集 → 检索 → 指标 → 报告」串起来（评测里唯一含 I/O 的代码）。

与 `evaluation.py` 的分工：
- `evaluation.py`：**纯函数**（数据集模型、指标、报告、对比），无 I/O、可离线单测；
- 本模块：**编排层**，负责开数据库会话、调用线上检索链路（`retrieval.retrieve_hits`）、
  在需要时执行 Query Rewrite，再把结果喂给纯函数算出报告。

为什么单独一个文件：`tests/rag/` 的 pytest 与 `scripts/rag_eval.py` 的 CLI 必须共用同一套编排，
否则「pytest 跑出来的数」与「CLI 跑出来的数」会不一致，基线对比就失去意义。

⚠️ **全程只读**：不写库、不改索引、不建会话、不落消息；临时改动的配置在 `override_settings()`
里成对还原（`get_settings()` 是 lru_cache 单例，测评完必须还原，否则会污染同一进程后续用例）。

⚠️ 为什么要临时放大 `rag_top_k`：Recall@1/3/5/10 需要同一份排序结果，若按线上 `rag_top_k=4`
检索就只能算 Recall@4。放大到 `EVAL_TOP_K` 后一次检索算出全部 K，且**排序与线上一致**
（Reranker 作用在整个候选池上，`rag_top_k` 只决定最后截断到几条）。
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.modules.ai.chat.messages import build_messages, chunk_text
from app.modules.ai.evaluation import (
    DEFAULT_KS,
    Category,
    EvalReport,
    GoldenDataset,
    GoldenFile,
    GoldenQuestion,
    abstain_pass,
    answer_correctness,
    citation_metrics,
    config_snapshot,
    group_by_category,
    macro_average,
    points_supported_by_context,
    retrieval_scores,
)
from app.modules.ai.grounding import (
    EvidenceReason,
    abstain_message,
    assess_evidence,
    grounding_prompt_enabled,
    needs_no_evidence_note,
    should_abstain,
)
from app.modules.ai.query import apply_query_rewrite, build_rewrite_llm

# 评测时临时使用的 top_k（一次算出 Recall@1/3/5/10）
EVAL_TOP_K = 10

# 评测用的伪会话标识（opencode 网关要求 `x-opencode-session` 非空；线上取真实会话 ID）
EVAL_SESSION_ID = "rag-eval"

# 每道题在报告里保留的命中明细条数（便于定位「换回了哪个文档」这类回归）
HIT_DETAIL_LIMIT = 10

# `matched_evidence` / `missing_evidence` 的统计口径（只看前 K 条）。
# 默认与 headline 指标 hit@5 对齐：否则会出现「missing 为空、hit@5 却是 0」的误读
# （证据确实召回了，但排在 6~10 名）。
DETAIL_K = 5


@contextmanager
def override_settings(**values: Any) -> Iterator[None]:
    """临时覆盖全局配置，退出时成对还原（`get_settings()` 是 lru_cache 单例）。"""
    settings = get_settings()
    saved = {key: getattr(settings, key) for key in values}
    try:
        for key, value in values.items():
            setattr(settings, key, value)
        yield
    finally:
        for key, value in saved.items():
            setattr(settings, key, value)


def git_state() -> dict[str, Any]:
    """记录当前代码状态（commit + 是否有未提交改动），让基线可解释、可复现。

    非 git 环境或命令失败时返回空 dict（评测不因此失败）。
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - 环境相关
        return {}
    return {"commit": commit, "dirty": dirty}


def _gateway_headers(base_url: str | None, session: str = EVAL_SESSION_ID) -> dict[str, str] | None:
    """opencode 网关要求 `x-opencode-session` **非空**，否则 400 `MissingSessionID`。

    线上由会话 ID 提供；评测没有会话，用一个固定标识（实测缺这个头会让
    Query Rewrite 直接降级为原问题 —— 而日志里只看到「回退原问题」，很容易误判成"改写没用"）。
    """
    return {"x-opencode-session": session} if "opencode" in (base_url or "") else None


async def build_eval_rewrite_llm(db: AsyncSession) -> Any | None:
    """构造 Query Rewrite 专用客户端（仅在 `rag_query_rewrite_enabled` 打开时才需要）。

    评测不依赖 Agent，所以这里直接取一个启用的 chat 模型实例（与线上同一套供应商参数）；
    找不到或构造失败时返回 None —— `apply_query_rewrite()` 会自行降级为「用原问题检索」。
    """
    if not get_settings().rag_query_rewrite_enabled:
        return None
    row = (
        await db.execute(
            text(
                "SELECT i.code, p.api_key, p.base_url FROM sys_model_instance i "
                "JOIN sys_model_provider p ON p.id = i.provider_id "
                "WHERE i.status = 1 AND p.status = 1 AND i.model_type = 'chat' "
                "ORDER BY i.create_time LIMIT 1"
            )
        )
    ).first()
    if row is None:
        return None
    return build_rewrite_llm(
        row[0], api_key=row[1], base_url=row[2], default_headers=_gateway_headers(row[2])
    )


async def build_eval_chat_llm(db: AsyncSession) -> Any | None:
    """构造答案层评测用的 chat 客户端（**temperature=0** 提高可复现性）。

    ⚠️ 与线上的两处刻意差异：
    1. 直接取一个启用的 chat 模型实例，**不复用 Agent 配置快照** —— 评测不该依赖
       「Agent 当前版本是否已发布」，否则改个草稿就会让结果变化；
    2. `temperature=0` —— 评测要可比较；线上用 Agent 配置（默认 0.7）。

    但检索、提示词组装、引用审计**全部复用线上同一套函数**，指标口径与线上一致。
    opencode 网关要求 `x-opencode-session` 非空（否则 400 MissingSessionID），故这里固定一个标识。
    """
    row = (
        await db.execute(
            text(
                "SELECT i.code, p.api_key, p.base_url FROM sys_model_instance i "
                "JOIN sys_model_provider p ON p.id = i.provider_id "
                "WHERE i.status = 1 AND p.status = 1 AND i.model_type = 'chat' "
                "ORDER BY i.create_time LIMIT 1"
            )
        )
    ).first()
    if row is None:
        return None

    from app.modules.ai.reasoning import ReasoningChatOpenAI

    settings = get_settings()
    base_url = row[2] or ""
    return ReasoningChatOpenAI(
        model=row[0],
        api_key=row[1] or "not-set",
        base_url=base_url or None,
        default_headers=_gateway_headers(base_url),
        temperature=0.0,
        # 与 AgentConfig 的默认 max_tokens 保持一致
        max_tokens=2048,
        max_retries=settings.ai_max_retries,
        timeout=settings.ai_request_timeout,
    )


async def retrieve_for_question(
    db: AsyncSession, *, knowledge_base_id: str, query: str, top_k: int = EVAL_TOP_K
) -> list[dict]:
    """按线上同配置检索（只放大 top_k），返回排序后的命中列表。"""
    from app.modules.ai.retrieval import retrieve_hits

    with override_settings(rag_top_k=top_k):
        return await retrieve_hits(db, {"knowledge_ids": [knowledge_base_id]}, query)


def _hit_detail(hits: Sequence[Mapping[str, Any]], expected_evidence: Sequence[str]) -> list[dict]:
    """命中的精简明细（排名 + 文档名 + 分数 + 该片段命中了哪几条证据）。"""
    detail: list[dict] = []
    for rank, hit in enumerate(hits[:HIT_DETAIL_LIMIT], start=1):
        content = str(hit.get("content") or "")
        detail.append(
            {
                "rank": rank,
                "document_name": hit.get("document_name") or "",
                "chunk_id": hit.get("chunk_id"),
                "score": hit.get("score"),
                "rerank_score": hit.get("rerank_score"),
                "matched_evidence": [item for item in expected_evidence if item in content],
            }
        )
    return detail


async def _two_pass_scores(
    db: AsyncSession,
    dataset: GoldenDataset,
    question: GoldenQuestion,
    ks: Sequence[int],
    *,
    top_k: int,
    rewrite_llm: Any | None,
    primary_hits: list[dict],
) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """返回（线上路径得分, 原始问题得分）。

    无历史的题两道 pass 完全等价（改写的前提是有上下文），因此只跑一次、结果复用，
    省掉一半检索开销（每次检索都包含一次 embedding 调用）。
    """
    production = retrieval_scores(primary_hits, question, ks)
    if not question.history:
        return production, production
    raw_hits = await retrieve_for_question(
        db, knowledge_base_id=dataset.knowledge_base_id, query=question.question, top_k=top_k
    )
    return production, retrieval_scores(raw_hits, question, ks)


async def run_retrieval_eval(
    db: AsyncSession,
    golden: GoldenFile,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    tag: str = "",
    limit: int | None = None,
    categories: Sequence[str] | None = None,
    top_k: int = EVAL_TOP_K,
    dataset_path: str | Path | None = None,
    detail_k: int = DETAIL_K,
    kb_override: str | None = None,
) -> EvalReport:
    """跑检索层评测，返回完整报告（可直接 `to_dict()` 落 JSON）。

    - `limit` / `categories` 用于快速子集调试（例如只跑 cross_chunk 的 5 道题）；
    - `no_answer` 题不参与 Recall/MRR（没有证据可算），单独统计「是否真的没命中」；
    - 线上路径会按配置决定是否执行 Query Rewrite，同时总保留一份「原始问题」口径用于对照；
    - `detail_k`：逐题 `matched_evidence` / `missing_evidence` 只看前 K 条命中（默认 5，
      与 headline 指标 hit@5 对齐）；报告里有 `detail_k` 字段说明取值。
    """
    rewrite_llm = await build_eval_rewrite_llm(db)
    selected = _select_questions(golden, limit=limit, categories=categories)

    rows: list[dict[str, Any]] = []
    for dataset, question in selected:
        kb_id = kb_override or dataset.knowledge_base_id
        hits: list[dict] = []
        if not question.is_no_answer:
            rewritten = await apply_query_rewrite(rewrite_llm, question.question, list(question.history))
            hits = await retrieve_for_question(
                db, knowledge_base_id=kb_id, query=rewritten.rewritten_query, top_k=top_k
            )
        assessment = assess_evidence(hits)
        production, raw = await _two_pass_scores(
            db, dataset, question, ks, top_k=top_k, rewrite_llm=rewrite_llm, primary_hits=hits
        )
        matched = [
            item
            for item in question.expected_evidence
            if any(item in str(hit.get("content") or "") for hit in hits[:detail_k])
        ]
        rows.append(
            {
                "id": question.id,
                "category": question.category,
                "dataset": dataset.name,
                "question": question.question,
                "abstain_expected": question.is_no_answer,
                "retrieval": production,
                "retrieval_raw": raw,
                "hit_count": len(hits),
                "no_hits": not hits,
                "reason": assessment.reason,
                "low_relevance": assessment.reason == EvidenceReason.LOW_RELEVANCE.value,
                "detail_k": detail_k,
                "matched_evidence": matched,
                "missing_evidence": [item for item in question.expected_evidence if item not in matched],
                "hits": _hit_detail(hits, question.expected_evidence),
            }
        )

    return _build_report(
        rows,
        golden=golden,
        ks=ks,
        tag=tag,
        top_k=top_k,
        dataset_path=dataset_path,
    )


def _select_questions(
    golden: GoldenFile, *, limit: int | None, categories: Sequence[str] | None
) -> list[tuple[GoldenDataset, GoldenQuestion]]:
    """按数据集顺序取题，可选按分类过滤与总量截断。"""
    wanted = set(categories) if categories else None
    selected: list[tuple[GoldenDataset, GoldenQuestion]] = []
    for dataset in golden.datasets:
        for question in dataset.questions:
            if wanted is not None and question.category not in wanted:
                continue
            selected.append((dataset, question))
            if limit is not None and len(selected) >= limit:
                return selected
    return selected


async def run_answers_eval(
    db: AsyncSession,
    golden: GoldenFile,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    tag: str = "",
    limit: int | None = None,
    categories: Sequence[str] | None = None,
    top_k: int = EVAL_TOP_K,
    dataset_path: str | Path | None = None,
    kb_override: str | None = None,
    llm: Any | None = None,
) -> EvalReport:
    """答案层评测：自编排完整回答链路，并产出 correctness / 引用 / 拒答 三类指标。

    链路与线上一致（共用同一套函数）：检索（含可选改写）→ `grounding` 判定 →
    `build_messages(grounding=...)` → 调用模型；**拒答时直接短路，不调用模型**。

    与线上的刻意差异：不走 Agent 工具模式、`temperature=0`、**全程不写库**
    （不建会话、不落消息、不写 usage）—— 因此可重复且不影响线上数据。
    报告同时带上检索层指标（复用同一次检索结果），便于一层运行看完两层；
    但**不做 raw 对照**（那是检索层评测的职责，避免双倍检索开销）。
    """
    resolved_llm = llm or await build_eval_chat_llm(db)
    if resolved_llm is None:
        raise RuntimeError("没有可用的 chat 模型实例，无法跑答案层评测")
    rewrite_llm = await build_eval_rewrite_llm(db)
    selected = _select_questions(golden, limit=limit, categories=categories)

    rows: list[dict[str, Any]] = []
    for dataset, question in selected:
        kb_id = kb_override or dataset.knowledge_base_id
        config = {"knowledge_ids": [kb_id]}
        rewritten = await apply_query_rewrite(rewrite_llm, question.question, list(question.history))
        hits = await retrieve_for_question(
            db, knowledge_base_id=kb_id, query=rewritten.rewritten_query, top_k=top_k
        )
        abstained = should_abstain(hits, config)
        latency_ms: int | None = None
        if abstained:
            answer = abstain_message()
        else:
            from app.modules.ai.retrieval import format_context

            messages = build_messages(
                "",
                format_context(hits),
                list(question.history),
                question.question,
                grounding=grounding_prompt_enabled(config),
                no_evidence_note=needs_no_evidence_note(hits, config),
            )
            started = time.perf_counter()
            response = await resolved_llm.ainvoke(messages, timeout=get_settings().ai_request_timeout)
            latency_ms = round((time.perf_counter() - started) * 1000)
            answer = chunk_text(response)

        sources = _sources_of(hits)
        citations = citation_metrics(answer, sources)
        points = question.answer_points()
        # 无答案题的金标"答案"就是拒答文案：correctness 仍算（能看出"答了但没按期望拒答"），
        # 但**不做 context 支撑校验** —— 拒答文案本来就不该出现在检索到的资料里，比对无意义且会污染指标。
        context_points = () if question.is_no_answer else points
        rows.append(
            {
                "id": question.id,
                "category": question.category,
                "dataset": dataset.name,
                "question": question.question,
                "expected_answer": question.expected_answer,
                "abstain_expected": question.is_no_answer,
                "retrieval": retrieval_scores(hits, question, ks),
                "retrieval_raw": None,
                "hit_count": len(hits),
                "no_hits": not hits,
                "reason": assess_evidence(hits).reason,
                "low_relevance": assess_evidence(hits).reason == EvidenceReason.LOW_RELEVANCE.value,
                "matched_evidence": [
                    item
                    for item in question.expected_evidence
                    if any(item in str(hit.get("content") or "") for hit in hits[:DETAIL_K])
                ],
                "missing_evidence": [],
                "detail_k": DETAIL_K,
                "hits": _hit_detail(hits, question.expected_evidence),
                "answer": {
                    "answer": answer,
                    "abstained": abstained,
                    "abstain_expected": question.is_no_answer,
                    "abstain_pass": (
                        abstain_pass(answer, abstained=abstained) if question.is_no_answer else None
                    ),
                    "correctness": answer_correctness(answer, points),
                    "supported_by_context": points_supported_by_context(
                        answer, context_points, format_context(hits)
                    ),
                    "citations": {
                        "cited": list(citations.cited),
                        "invalid_cited": list(citations.invalid_cited),
                        "uncited_claims": citations.uncited_claims,
                        "has_citation": citations.has_citation,
                    },
                    "latency_ms": latency_ms,
                },
            }
        )

    return _build_report(
        rows,
        golden=golden,
        ks=ks,
        tag=tag,
        top_k=top_k,
        dataset_path=dataset_path,
        answers=_build_answers(rows),
    )


def _sources_of(hits: Sequence[Mapping[str, Any]]) -> list[dict]:
    """把命中片段映射成 `AiSourceOut` 形状（引用审计需要 `index` 与来源列表对应）。"""
    from app.modules.ai.retrieval import build_sources

    return build_sources(list(hits))


def _answer_metric_row(record: Mapping[str, Any]) -> dict[str, float | None]:
    """单题的答案层指标行（交给 `macro_average()` 聚合）。"""
    answer = record["answer"]
    citations = answer["citations"]
    return {
        "answer_correctness": answer["correctness"],
        "points_supported_rate": answer["supported_by_context"],
        "citation_valid_rate": 0.0 if citations["invalid_cited"] else 1.0,
        "cited_present_rate": 1.0 if citations["has_citation"] else 0.0,
        "abstain_pass": answer["abstain_pass"],
        "latency_ms": float(answer["latency_ms"]) if answer["latency_ms"] is not None else None,
    }


def _build_answers(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """答案层报告：整体 + 分类。

    两个派生指标的语义（容易被误读，这里写清楚）：
    - `abstain_accuracy`：无答案题的通过率（宽松口径：ABSTAIN 或正文明确声明无法确认）；
    - `false_abstain_rate`：**有答案题**被拒答的比例（越低越好，高说明阈值过严）；
    - `uncited_claims_total`：全部题的无引用事实句总数（越低越好）。
    """
    by_category: dict[str, dict[str, float]] = {}
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for record in rows:
        buckets.setdefault(str(record["category"]), []).append(record)
    for category, bucket in sorted(buckets.items()):
        by_category[category] = _answers_overall(bucket)
    return {"overall": _answers_overall(rows), "by_category": by_category}


def _answers_overall(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """答案层整体指标（macro；无样本的指标不出现）。"""
    metrics = macro_average([_answer_metric_row(record) for record in rows])
    answerable = [record for record in rows if not record["abstain_expected"]]
    overall = {key: value for key, value in metrics.items() if key != "abstain_pass"}
    if "abstain_pass" in metrics:
        overall["abstain_accuracy"] = metrics["abstain_pass"]
    if answerable:
        overall["false_abstain_rate"] = sum(
            1 for record in answerable if record["answer"]["abstained"]
        ) / len(answerable)
    overall["uncited_claims_total"] = float(
        sum(int(record["answer"]["citations"]["uncited_claims"]) for record in rows)
    )
    return overall


def _build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    golden: GoldenFile,
    ks: Sequence[int],
    tag: str,
    top_k: int,
    dataset_path: str | Path | None,
    answers: dict[str, Any] | None = None,
) -> EvalReport:
    """把逐题结果汇总成报告（整体 / 分类 / 无答案题统计）。"""
    production_rows = [row["retrieval"] for row in rows if row.get("retrieval")]
    raw_rows = [row["retrieval_raw"] for row in rows if row.get("retrieval_raw")]
    return EvalReport(
        meta={
            "ts": datetime.now(UTC).isoformat(),
            "tag": tag,
            "ks": list(ks),
            "eval_top_k": top_k,
            "questions": len(rows),
            "thresholds": dict(golden.thresholds),
            "dataset_path": str(dataset_path) if dataset_path else "",
            "dataset_version": golden.version,
            "config": config_snapshot(),
            "git": git_state(),
        },
        retrieval={
            "overall": macro_average(production_rows),
            "by_category": group_by_category(rows),
            "raw_overall": macro_average(raw_rows),
            "raw_by_category": group_by_category(rows, metric_prefix="retrieval_raw"),
            "no_answer": _no_answer_stats(rows),
        },
        answers=answers,
        per_question=[dict(row) for row in rows],
    )


def _no_answer_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """无答案题的检索侧观测（不设门禁，只回答「它们真的检索不到东西吗」）。"""
    bucket = [row for row in rows if row["category"] == Category.NO_ANSWER.value]
    if not bucket:
        return {}
    total = len(bucket)
    return {
        "count": float(total),
        "no_hits_rate": sum(1 for row in bucket if row["no_hits"]) / total,
        "low_relevance_rate": sum(1 for row in bucket if row["low_relevance"]) / total,
    }
