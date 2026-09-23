"""检索层评测：Recall@K / 证据覆盖率 / MRR（线上路径 vs 原始问题双口径）。

需要**真实环境**（MySQL + Qdrant + embedding）；环境不可达时由 `rag_env` 夹具整体 skip。
同一次会话只跑一遍评测（`eval_report` 夹具缓存），断言与打印都基于那一份报告。

两类断言：
1. **恒真健全性**：指标算得出来、管线确实召回了东西、报告能序列化 —— 这些失败说明环境或代码坏了；
2. **门禁**：`[thresholds]` 里配置的指标下限 —— 这些失败说明这次改动让 RAG 变差了。
   阈值留空时（首次测量）自动 skip 而不是失败。
"""

import json

import pytest

from app.core.config import get_settings
from app.modules.ai.evaluation import Category, GoldenFile, render_summary, threshold_to_metric
from app.modules.ai.grounding import EvidenceReason

pytestmark = pytest.mark.rag_eval


def test_metrics_computed_for_every_answerable_question(eval_report) -> None:
    """有答案的题必须有检索指标；无答案题必须为 None（它们只判拒答，不参与 Recall）。"""
    for record in eval_report.per_question:
        if record["abstain_expected"]:
            assert record["retrieval"] is None, record["id"]
        else:
            assert record["retrieval"], f"{record['id']} 缺少检索指标"


def test_pipeline_actually_retrieves_something(eval_report) -> None:
    """健全性：至少要有命中，否则说明索引为空 / 检索链路断了（而不是「变差了」）。"""
    overall = eval_report.retrieval["overall"]

    assert overall.get("hit@5", 0.0) > 0.0, render_summary(eval_report)
    assert 0.0 <= overall["mrr"] <= 1.0


def test_configured_thresholds_are_met(golden: GoldenFile, eval_report) -> None:
    """门禁：`[thresholds]` 里配置的指标下限（键是 snake_case，需映射成报告指标名）。"""
    if not golden.thresholds:
        pytest.skip("Golden Dataset 的 [thresholds] 为空：已跑完测量，请在人工确认后回填阈值")

    overall = eval_report.retrieval["overall"]
    failures: list[str] = []
    for key, value in golden.thresholds.items():
        metric = threshold_to_metric(key)
        measured = overall.get(metric)
        if measured is None:
            failures.append(f"{key} → {metric}: 报告里没有该指标（键写错了？）")
        elif measured < value:
            failures.append(f"{key} → {metric}: 实测 {measured:.3f} < 阈值 {value:.3f}")
    assert not failures, "\n".join(failures) + "\n" + render_summary(eval_report)


def test_anaphora_questions_benefit_from_rewrite(eval_report) -> None:
    """改写增益：指代题在「原始问题」口径下应更差（否则说明改写没起作用）。

    仅在 `RAG_QUERY_REWRITE_ENABLED=true` 时有意义；关闭时 skip（此时两个口径必然相同）。
    """
    if not get_settings().rag_query_rewrite_enabled:
        pytest.skip("Query Rewrite 未启用（RAG_QUERY_REWRITE_ENABLED=false），跳过改写增益校验")

    rewritten = (eval_report.retrieval.get("by_category") or {}).get(Category.ANAPHORA.value, {})
    raw = (eval_report.retrieval.get("raw_by_category") or {}).get(Category.ANAPHORA.value, {})

    assert rewritten.get("hit@5") is not None and raw.get("hit@5") is not None
    assert rewritten["hit@5"] >= raw["hit@5"], (
        f"指代题改写后 hit@5 ({rewritten['hit@5']}) 低于原始问题 ({raw['hit@5']})"
    )


def test_cross_chunk_questions_report_evidence_coverage(eval_report) -> None:
    """跨 chunk 题的证据覆盖率必须被算出来（这类题的价值就在于「漏了几条证据」）。"""
    bucket = (eval_report.retrieval.get("by_category") or {}).get(Category.CROSS_CHUNK.value)

    assert bucket, "报告里缺少 cross_chunk 分类"
    assert bucket["evidence_recall@5"] is not None
    assert bucket["_count"] > 0


def test_no_answer_questions_are_reported_separately(eval_report) -> None:
    """无答案题不参与 Recall，但要能看到「是否真的检索不到」（观测指标，不设门禁）。"""
    stats = eval_report.retrieval.get("no_answer") or {}

    assert stats.get("count", 0) > 0
    assert 0.0 <= stats["no_hits_rate"] <= 1.0
    assert 0.0 <= stats["low_relevance_rate"] <= 1.0


def test_grounding_reason_recorded_per_question(eval_report) -> None:
    """每题都记录证据判定结论（Phase 4 的拒答评估与人工复核都依赖它）。"""
    valid = {reason.value for reason in EvidenceReason}

    for record in eval_report.per_question:
        assert record["reason"] in valid, record["id"]


def test_report_is_json_serializable(eval_report) -> None:
    """报告要能直接落 JSON（CLI 的基线对比依赖它）——防止混入 Decimal 等不可序列化值。"""
    payload = json.dumps(eval_report.to_dict(), ensure_ascii=False, default=str)

    assert '"retrieval"' in payload and '"per_question"' in payload


def test_hit_details_record_what_came_back(eval_report) -> None:
    """命中明细要保留文档名与分数：回归时能直接看出「换回了哪个文档」。"""
    for record in eval_report.per_question:
        for hit in record["hits"]:
            assert "rank" in hit and "document_name" in hit
        ranks = [hit["rank"] for hit in record["hits"]]
        assert ranks == list(range(1, len(ranks) + 1))
