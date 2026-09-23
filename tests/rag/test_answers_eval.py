"""答案层评测（**会真实调用模型**，花 token；默认 skip）。

启用方式：
    RAG_EVAL_ANSWERS=1 .venv/bin/pytest tests/rag/ -q          # 默认跑 10 题（便宜）
    RAG_EVAL_ANSWERS=1 RAG_EVAL_ANSWER_LIMIT=0 .venv/bin/pytest tests/rag/ -q   # 跑全部 40 题

断言原则（重要）：**不对模型输出内容设硬门禁**。大模型本身有随机性，把「必须答出某句话」
写成断言会造出一堆 flaky 用例，最后被大家无视。这里只断言：
1. 指标算得出来、结构完整（确定性）；
2. 链路行为正确 —— 尤其「拒答时不调用模型」（可由 `latency_ms is None` 反证）；
3. 复核表能生成、能被解析回来。

回答质量好不好，交给人工复核表 + `scripts/rag_eval.py --baseline` 的基线对比判断。
"""

import json

import pytest

from app.modules.ai.evaluation import (
    REVIEW_VERDICT_COLUMN,
    parse_review_markdown,
    render_review_markdown,
    summarize_review,
)

pytestmark = pytest.mark.rag_eval

# 答案层指标必须齐全的键（缺一个就说明聚合逻辑漏了）
REQUIRED_OVERALL_KEYS = {
    "answer_correctness",
    "citation_valid_rate",
    "cited_present_rate",
    "uncited_claims_total",
    "false_abstain_rate",
}


def test_answer_metrics_are_present(answers_report) -> None:
    overall = (answers_report.answers or {}).get("overall") or {}

    missing = REQUIRED_OVERALL_KEYS - set(overall)
    assert not missing, f"答案层指标缺失: {sorted(missing)}（实际: {sorted(overall)}）"
    assert 0.0 <= overall["citation_valid_rate"] <= 1.0
    assert 0.0 <= overall["false_abstain_rate"] <= 1.0


def test_by_category_metrics_are_grouped(answers_report) -> None:
    by_category = (answers_report.answers or {}).get("by_category") or {}

    assert by_category, "答案层缺少分类汇总"
    for category, stats in by_category.items():
        assert "answer_correctness" in stats or "citation_valid_rate" in stats, category


def test_abstain_short_circuit_skips_model_call(answers_report) -> None:
    """拒答必须走短路：**没有 latency** 就说明没调用模型（确定性、可硬断言）。

    这条是 Phase 3 那条「拒答不调用 LLM」结论在答案层的等价校验；若哪天有人在拒答分支
    里加了模型调用（比如让模型改写拒答文案），这里会立刻红。
    """
    answered = [record for record in answers_report.per_question if not record["answer"]["abstained"]]
    abstained = [record for record in answers_report.per_question if record["answer"]["abstained"]]

    for record in abstained:
        assert record["answer"]["latency_ms"] is None, record["id"]
    for record in answered:
        assert record["answer"]["latency_ms"] is not None, record["id"]


def test_correctness_computed_for_every_question(answers_report) -> None:
    """每题都要有要点命中分 —— 包括无答案题。

    无答案题的金标答案就是拒答文案本身（数据集里写明），因此它也参与 correctness：
    「答了但没按期望拒答」会体现为分数 < 1，比只给一个布尔更有信息量。
    """
    for record in answers_report.per_question:
        assert record["answer"]["correctness"] is not None, record["id"]


def test_context_support_follows_its_definition(answers_report) -> None:
    """`supported_by_context` 是**有定义域**的指标，这里断言它的不变式：

    - 无答案题：恒为 None（拒答文案不该出现在资料里，比对无意义）；
    - 有答案题：只有当答案命中了金标要点时才有值；否则为 None
      —— `None` 意味着「答非所问 / 没答到要点」，**不是**「没有幻觉」。
    """
    for record in answers_report.per_question:
        answer = record["answer"]
        if record["abstain_expected"]:
            assert answer["supported_by_context"] is None, record["id"]
            continue
        assert (answer["supported_by_context"] is None) == (answer["correctness"] == 0.0), record["id"]


def test_uncited_claims_are_recorded_per_question(answers_report) -> None:
    """引用审计必须逐题留痕（人工复核时要看「哪句没标来源」）。"""
    for record in answers_report.per_question:
        citations = record["answer"]["citations"]
        assert isinstance(citations["uncited_claims"], int)
        assert isinstance(citations["invalid_cited"], list)
        assert isinstance(citations["cited"], list)


def test_no_answer_questions_reported_in_answers_layer(answers_report) -> None:
    """无答案题在答案层只判拒答（宽松口径）；没有这类题时本用例无意义，直接放过。"""
    no_answer = [record for record in answers_report.per_question if record["abstain_expected"]]
    if not no_answer:
        pytest.skip("本次答案层评测没有抽到无答案题（由 RAG_EVAL_ANSWER_LIMIT 决定）")

    accuracies = {bool(record["answer"]["abstain_pass"]) for record in no_answer}
    assert accuracies  # 至少要有布尔值被算出来
    overall = (answers_report.answers or {}).get("overall") or {}
    assert 0.0 <= overall["abstain_accuracy"] <= 1.0


def test_review_sheet_renders_and_parses(answers_report) -> None:
    """复核表：能生成（列齐、行数=题数），未填写时解析为空，模拟填写后能读回来。"""
    sheet = render_review_markdown(answers_report)
    data_rows = [line for line in sheet.splitlines() if line.startswith("| ") and "---" not in line]

    assert REVIEW_VERDICT_COLUMN in sheet
    assert len(data_rows) == len(answers_report.per_question) + 1  # + 表头
    assert parse_review_markdown(sheet) == {}  # 人工结论列还是空的

    filled = sheet.replace(" |  |  |", " | 通过 |  |", 1)
    verdicts = parse_review_markdown(filled)
    if verdicts:  # 只有成功替换到一行才会解析出结果
        assert summarize_review(verdicts)["pass"] == 1
        assert summarize_review(verdicts)["pass_rate"] == 1.0


def test_answers_report_is_json_serializable(answers_report) -> None:
    payload = json.dumps(answers_report.to_dict(), ensure_ascii=False, default=str)

    assert '"answers"' in payload and '"abstained"' in payload
