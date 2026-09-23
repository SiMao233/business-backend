"""RAG 评测内核的离线单测：数据校验 / 检索指标 / 答案指标 / 聚合 / 报告对比。

**不需要任何外部环境**（不连 MySQL、Qdrant、模型网关）—— 全部断言落在纯函数上，
因此这个文件可以在任何机器上随时跑（也是 CI 友好的一组）。
"""

import tomllib
from pathlib import Path

import pytest

from app.modules.ai.evaluation import (
    REVIEW_VERDICT_COLUMN,
    Category,
    DatasetError,
    EvalReport,
    ExpectedBehavior,
    GoldenQuestion,
    abstain_pass,
    answer_correctness,
    citation_metrics,
    compare_metrics,
    compare_reports,
    config_snapshot,
    evidence_matches,
    evidence_recall_at_k,
    first_evidence_rank,
    group_by_category,
    hit_at_k,
    known_threshold_keys,
    load_golden_file,
    macro_average,
    parse_golden,
    parse_review_markdown,
    points_supported_by_context,
    reciprocal_rank,
    render_diff,
    render_markdown_table,
    render_review_markdown,
    render_summary,
    retrieval_scores,
    source_recall_at_k,
    summarize_review,
    threshold_to_metric,
)

# ---- 夹具与构造助手 ----

TOML_VALID = """
version = 1

[thresholds]
hit_at_5 = 0.6
mrr = 0.45

[[datasets]]
name = "面试八股文"
knowledge_base_id = "e730c993415742c48733e4d526ff9959"

[[datasets.questions]]
id = "fact-01"
category = "simple_fact"
question = "员工手册规定的标准工作时间是几点到几点？"
expected_answer = "周一至周五上午九点到下午六点，午休一小时"
expected_answer_points = ["上午九点", "下午六点"]
expected_source = ["员工手册.txt"]
expected_evidence = ["工作时间为周一至周五上午九点到下午六点"]

[[datasets.questions]]
id = "na-01"
category = "no_answer"
question = "公司年会的具体安排是什么？"
expected_behavior = "abstain"
notes = "知识库里确实没有"
"""


def _toml_question(**overrides: object) -> str:
    """基一份合法单题 TOML；override 值均为**已格式化的 TOML 片段**（如 `'"nope"'` / `'["a.md"]'`）。

    值为 None 表示删掉该行（用于构造「缺字段」的非法 case）。
    """
    fields: dict[str, object] = {
        "id": '"q-01"',
        "category": '"simple_fact"',
        "question": '"怎么配置？"',
        "expected_answer": '"在配置文件中设置"',
        "expected_source": '["a.md"]',
        "expected_evidence": '["配置文件中设置"]',
    }
    fields.update(overrides)
    lines = [f"{key} = {value}" for key, value in fields.items() if value is not None]
    header = (
        'version = 1\n\n[[datasets]]\nname = "kb"\n'
        'knowledge_base_id = "abc"\n\n[[datasets.questions]]\n'
    )
    return header + "\n".join(lines) + "\n"


def _expect_dataset_error(toml_text: str, *fragments: str) -> None:
    """断言校验失败，并检查报错里包含可定位的字段级原因。"""
    with pytest.raises(DatasetError) as excinfo:
        parse_golden(tomllib.loads(toml_text))
    message = str(excinfo.value)
    for fragment in fragments:
        assert fragment in message, f"报错缺少片段 {fragment!r}：\n{message}"


def _hit(content: str, document_name: str = "员工手册.txt", score: float = 0.5) -> dict:
    """构造一条检索命中（形状与 retrieve_hits 产出一致）。"""
    return {"content": content, "document_name": document_name, "score": score}


def _question(**overrides: object) -> GoldenQuestion:
    fields: dict[str, object] = {
        "id": "q-01",
        "category": Category.SIMPLE_FACT.value,
        "question": "问题",
        "expected_answer": "答案",
        "expected_source": ("员工手册.txt",),
        "expected_evidence": ("标准工作时间",),
    }
    fields.update(overrides)
    return GoldenQuestion(**fields)  # type: ignore[arg-type]


# ---- 一、数据集加载与校验 ----


def test_load_golden_file_reads_valid_toml(tmp_path: Path) -> None:
    path = tmp_path / "golden.toml"
    path.write_text(TOML_VALID, encoding="utf-8")

    golden = load_golden_file(path)

    assert golden.version == 1
    assert golden.thresholds == {"hit_at_5": 0.6, "mrr": 0.45}
    assert [question.id for question in golden.questions] == ["fact-01", "na-01"]
    assert golden.categories() == {"simple_fact": 1, "no_answer": 1}


def test_load_golden_file_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="数据集文件不存在"):
        load_golden_file(tmp_path / "nope.toml")


def test_no_answer_question_excluded_from_retrieval_metrics() -> None:
    """无答案题没有 evidence → 不参与检索指标（否则 Recall 会被真空真值污染）。"""
    question = parse_golden(tomllib.loads(TOML_VALID)).questions[1]

    assert question.is_no_answer is True
    assert retrieval_scores([_hit("任意内容")], question) is None


def test_answer_points_fall_back_to_expected_answer() -> None:
    with_points = _question(expected_answer_points=("要点",))
    without_points = _question(expected_answer="整段答案")

    assert with_points.answer_points() == ("要点",)
    assert without_points.answer_points() == ("整段答案",)


def test_validation_rejects_missing_evidence() -> None:
    _expect_dataset_error(_toml_question(expected_evidence=None), "expected_evidence 不能为空")


def test_validation_rejects_missing_source() -> None:
    _expect_dataset_error(_toml_question(expected_source=None), "expected_source 不能为空")


def test_validation_rejects_missing_question_text() -> None:
    _expect_dataset_error(_toml_question(question='""'), "question 不能为空")


def test_validation_rejects_duplicate_ids() -> None:
    text = _toml_question() + """
[[datasets.questions]]
id = "q-01"
category = "keyword"
question = "重复 id"
expected_answer = "x"
expected_source = ["a.md"]
expected_evidence = ["x"]
"""
    _expect_dataset_error(text, "id 重复: q-01")


def test_validation_rejects_unknown_keys() -> None:
    """字段拼错必须立刻失败，否则题目会静默变成永不命中的噪音。"""
    _expect_dataset_error(_toml_question(categor='"simple_fact"'), "含未知键")


def test_validation_rejects_invalid_category_and_behavior() -> None:
    _expect_dataset_error(_toml_question(category='"nope"'), "category 非法")
    _expect_dataset_error(_toml_question(expected_behavior='"maybe"'), "expected_behavior 非法")


def test_validation_requires_history_for_multi_turn_and_anaphora() -> None:
    _expect_dataset_error(_toml_question(category='"anaphora"'), "必须提供 history")
    _expect_dataset_error(_toml_question(category='"multi_turn"'), "必须提供 history")


def test_validation_accepts_history_pairs() -> None:
    text = _toml_question(
        category='"anaphora"',
        history='[["user", "退款期限是多少？"], ["assistant", "60 天。"]]',
    )

    question = parse_golden(tomllib.loads(text), source="mem").questions[0]

    assert question.history == (("user", "退款期限是多少？"), ("assistant", "60 天。"))


def test_validation_rejects_evidence_on_no_answer_question() -> None:
    text = _toml_question(
        category='"no_answer"',
        expected_behavior='"abstain"',
        expected_evidence='["不该有"]',
        expected_source=None,
    )
    _expect_dataset_error(text, "无答案题不应填 expected_evidence")


def test_validation_rejects_non_numeric_threshold() -> None:
    text = _toml_question().replace("version = 1", "version = 1\n[thresholds]\nhit_at_5 = \"high\"")
    _expect_dataset_error(text, "thresholds.hit_at_5 必须是数值")


# ---- 二、检索层指标 ----


def test_evidence_matches_keeps_declared_order() -> None:
    assert evidence_matches("甲和乙都在这里", ("乙", "甲")) == ("乙", "甲")
    assert evidence_matches("只有甲", ("乙",)) == ()


def test_first_evidence_rank_and_missing() -> None:
    hits = [_hit("无关"), _hit("这里出现标准工作时间"), _hit("再来一次标准工作时间")]

    assert first_evidence_rank(hits, ("标准工作时间",)) == 2
    assert first_evidence_rank(hits, ("根本不存在的句子",)) is None


def test_hit_at_k_boundary() -> None:
    """边界：排名恰好等于 K 算命中，K+1 不算。"""
    hits = [_hit("无关"), _hit("标准工作时间")]

    assert hit_at_k(hits, ("标准工作时间",), 2) == 1.0
    assert hit_at_k(hits, ("标准工作时间",), 1) == 0.0


def test_evidence_recall_partial_for_cross_chunk() -> None:
    """跨 chunk 题：证据分散在多条命中里，只覆盖一条时 Recall = 0.5。"""
    hits = [_hit("前一半证据"), _hit("完全无关")]

    assert evidence_recall_at_k(hits, ("前一半证据", "后一半证据"), 5) == 0.5
    assert evidence_recall_at_k([_hit("前一半证据后一半证据")], ("前一半证据", "后一半证据"), 1) == 1.0


def test_evidence_recall_without_expected_is_vacuous() -> None:
    assert evidence_recall_at_k([], (), 5) == 1.0


def test_source_recall_counts_documents() -> None:
    hits = [_hit("x", document_name="a.md"), _hit("y", document_name="b.md")]

    assert source_recall_at_k(hits, ("a.md", "b.md"), 5) == 1.0
    assert source_recall_at_k(hits, ("a.md", "c.md"), 5) == 0.5
    assert source_recall_at_k(hits, (), 5) == 1.0


def test_reciprocal_rank_values() -> None:
    assert reciprocal_rank([_hit("标准工作时间")], ("标准工作时间",)) == 1.0
    second_place = [_hit("无关"), _hit("无关"), _hit("标准工作时间")]
    assert reciprocal_rank(second_place, ("标准工作时间",)) == pytest.approx(1 / 3)
    assert reciprocal_rank([_hit("无关")], ("标准工作时间",)) == 0.0


def test_retrieval_scores_keys_follow_ks() -> None:
    scores = retrieval_scores([_hit("标准工作时间")], _question(), ks=(1, 5))

    assert scores is not None
    assert set(scores) == {
        "hit@1",
        "hit@5",
        "evidence_recall@1",
        "evidence_recall@5",
        "source_recall@1",
        "source_recall@5",
        "mrr",
    }
    assert scores["mrr"] == 1.0


# ---- 三、答案层指标 ----


def test_answer_correctness_partial_and_undefined() -> None:
    assert answer_correctness("上午九点上班，下午六点下班", ("上午九点", "下午六点")) == 1.0
    assert answer_correctness("上午九点上班", ("上午九点", "下午六点")) == 0.5
    assert answer_correctness("随便答的", ()) is None


def test_citation_metrics_reuse_grounding_audit() -> None:
    sources = [{"index": 1, "document_name": "员工手册.txt"}]

    valid = citation_metrics("标准工作时间是九点到六点 [1]。", sources)
    assert valid.cited == (1,)
    assert valid.invalid_cited == ()
    assert valid.valid is True
    assert valid.has_citation is True

    fabricated = citation_metrics("标准工作时间是九点到六点 [1]，另有规定 [9]。", sources)
    assert fabricated.invalid_cited == (9,)
    assert fabricated.valid is False


def test_points_supported_by_context_flags_unsupported_points() -> None:
    """答案写了要点、但本轮 context 里没有 → 疑似幻觉（分数低）。"""
    supported = points_supported_by_context("九点到六点", ("九点", "六点"), "工作时间为九点到六点")
    unsupported = points_supported_by_context("九点到六点", ("九点", "六点"), "完全无关的资料")

    assert supported == 1.0
    assert unsupported == 0.0
    assert points_supported_by_context("没答到要点", ("九点",), "九点") is None


def test_abstain_pass_accepts_hard_and_soft_refusal() -> None:
    assert abstain_pass("根据当前知识库，我无法确认这个问题。", abstained=True) is True
    assert abstain_pass("现有资料不足，无法确定具体安排。", abstained=False) is True
    assert abstain_pass("年会在下周五举行。", abstained=False) is False


# ---- 四、聚合 ----


def test_macro_average_skips_none() -> None:
    rows = [{"hit@5": 1.0, "mrr": 1.0}, {"hit@5": 0.0, "mrr": None}, {"hit@5": None, "mrr": 0.0}]

    assert macro_average(rows) == {"hit@5": 0.5, "mrr": 0.5}
    assert macro_average([]) == {}


def test_group_by_category_buckets_and_counts() -> None:
    per_question = [
        {"id": "a", "category": "simple_fact", "retrieval": {"hit@5": 1.0}},
        {"id": "b", "category": "simple_fact", "retrieval": {"hit@5": 0.0}},
        {"id": "c", "category": "no_answer", "retrieval": None},
    ]

    grouped = group_by_category(per_question)

    assert grouped["simple_fact"]["hit@5"] == 0.5
    assert grouped["simple_fact"]["_count"] == 2.0
    assert "no_answer" not in grouped  # retrieval 为 None 的题不进任何桶


def test_threshold_keys_map_to_metric_names() -> None:
    """阈值键是 snake_case（手写友好）、报告指标名带 `@`（可读）——映射必须显式且可测。

    没有这层映射时，门禁会拿 `hit_at_5` 去报告里取值拿到 None：
    要么静默失效，要么永远报错（实际就踩过一次）。
    """
    assert threshold_to_metric("hit_at_5") == "hit@5"
    assert threshold_to_metric("hit_at_10") == "hit@10"
    assert threshold_to_metric("evidence_recall_at_5") == "evidence_recall@5"
    assert threshold_to_metric("source_recall_at_3") == "source_recall@3"
    assert threshold_to_metric("mrr") == "mrr"
    assert threshold_to_metric("unknown_metric") == "unknown_metric"

    known = known_threshold_keys()
    assert {"hit_at_5", "evidence_recall_at_5", "source_recall_at_5", "mrr"} <= known
    assert "hit@5" not in known  # 报告风格的键不是合法阈值键（两面命名刻意不混用）


# ---- 五、报告与基线对比 ----


def _report(overall: dict[str, float], questions: list[dict], answers: dict | None = None) -> EvalReport:
    return EvalReport(
        meta={"tag": "t"},
        retrieval={"overall": overall, "by_category": {}},
        answers={"overall": answers} if answers else None,
        per_question=questions,
    )


def test_compare_metrics_direction_respects_metric_semantics() -> None:
    deltas = {
        delta.name: delta
        for delta in compare_metrics(
            {"hit@5": 0.8, "uncited_claims_total": 4.0, "mrr": 0.5},
            {"hit@5": 0.9, "uncited_claims_total": 1.0, "gone": 1.0},
        )
    }

    assert deltas["hit@5"].direction == "improved"  # 越高越好
    assert deltas["uncited_claims_total"].direction == "improved"  # 越低越好
    assert deltas["gone"].direction == "added"
    assert deltas["mrr"].direction == "removed"
    assert deltas["mrr"].delta is None


def test_compare_metrics_detects_regression_and_new_metric() -> None:
    deltas = {
        delta.name: delta for delta in compare_metrics({"hit@5": 0.9}, {"hit@5": 0.7, "hit@10": 0.95})
    }

    assert deltas["hit@5"].direction == "regressed"
    assert deltas["hit@10"].direction == "added"


def test_compare_metrics_unchanged_within_tolerance() -> None:
    deltas = {delta.name: delta for delta in compare_metrics({"hit@5": 0.5}, {"hit@5": 0.5 + 1e-12})}

    assert deltas["hit@5"].direction == "unchanged"


def test_compare_reports_lists_improved_and_regressed_questions() -> None:
    baseline = _report(
        {"hit@5": 0.5},
        [
            {"id": "a", "category": "simple_fact", "retrieval": {"hit@5": 1.0}},
            {"id": "b", "category": "anaphora", "retrieval": {"hit@5": 0.0}},
            {"id": "c", "category": "keyword", "retrieval": {"hit@5": 1.0}},
        ],
        answers={"correctness": 0.8},
    )
    current = _report(
        {"hit@5": 0.67},
        [
            {"id": "a", "category": "simple_fact", "retrieval": {"hit@5": 1.0}},
            {"id": "b", "category": "anaphora", "retrieval": {"hit@5": 1.0}},
            {"id": "d", "category": "keyword", "retrieval": {"hit@5": 1.0}},
        ],
        answers={"correctness": 0.5},
    )

    diff = compare_reports(baseline, current)

    assert [c.question_id for c in diff.improved_questions] == ["b"]
    assert diff.regressed_questions == ()
    assert diff.added_questions == ("d",)
    assert diff.removed_questions == ("c",)
    # 检索 hit@5 变好、但答案 correctness 变差 → 仍视为存在回退
    assert diff.has_regression is True


def test_report_roundtrip_and_renderers() -> None:
    report = _report({"hit@5": 0.5, "mrr": 0.4}, [{"id": "a", "category": "simple_fact", "retrieval": {"hit@5": 1.0}}])

    restored = EvalReport.from_dict(report.to_dict())
    assert restored.to_dict() == report.to_dict()

    summary = render_summary(report)
    assert "hit@5" in summary and "0.500" in summary

    diff_text = render_diff(compare_reports(report, _report({"hit@5": 0.9, "mrr": 0.4}, report.per_question)))
    assert "hit@5" in diff_text and "▲" in diff_text


def test_config_snapshot_covers_rag_switches() -> None:
    snapshot = config_snapshot()

    assert {"rag_top_k", "rag_hybrid_enabled", "rag_rerank_enabled", "rag_query_rewrite_enabled"} <= set(snapshot)


def test_expected_behavior_enum_values_are_stable() -> None:
    """报告与数据集里的字面量是外部契约，改动需显式（防手滑重命名）。"""
    assert ExpectedBehavior.ANSWER.value == "answer"
    assert ExpectedBehavior.ABSTAIN.value == "abstain"
    assert Category.NO_ANSWER.value == "no_answer"


# ---- 六、人工复核表 ----


def _answer_record(
    question_id: str,
    *,
    answer: str = "迟到超过三十分钟按半天事假处理 [1]。",
    invalid_cited: tuple[int, ...] = (),
    uncited_claims: int = 0,
    has_citation: bool = True,
) -> dict:
    """构造一条带答案层数据的逐题记录（形状与 run_answers_eval 产出一致）。"""
    return {
        "id": question_id,
        "category": "simple_fact",
        "question": "迟到怎么办？",
        "expected_answer": "按半天事假处理",
        "retrieval": {"hit@5": 1.0},
        "answer": {
            "answer": answer,
            "correctness": 1.0,
            "citations": {
                "cited": [1],
                "invalid_cited": list(invalid_cited),
                "uncited_claims": uncited_claims,
                "has_citation": has_citation,
            },
        },
    }


def test_render_markdown_table_pads_short_rows() -> None:
    """列数不匹配时必须补齐，而不是把表格撑坏。"""
    table = render_markdown_table(["A", "B", "C"], [["1", "2"]])
    rows = table.splitlines()

    assert rows[0] == "| A | B | C |"
    assert rows[1] == "| --- | --- | --- |"
    assert rows[2].count("|") == 4


def test_review_sheet_escapes_pipes_and_newlines() -> None:
    """单元格里的 `|` 与换行必须转义/压平，否则表格结构会崩（解析也会跟着错）。"""
    report = _report(
        {"hit@5": 1.0},
        [_answer_record("fact-01", answer="第一行|带竖线\n第二行")],
    )

    sheet = render_review_markdown(report)

    assert "\\|" in sheet
    assert "第一行|带竖线" not in sheet
    assert REVIEW_VERDICT_COLUMN in sheet
    assert parse_review_markdown(sheet) == {}  # 结构完好且人工列未填


def test_review_sheet_marks_fabricated_citations() -> None:
    """引用审计列要能一眼看出「模型编造了引用」。"""
    sheet = render_review_markdown(
        _report({"hit@5": 1.0}, [_answer_record("fact-01", invalid_cited=(9,))])
    )

    assert "有编造" in sheet


def test_parse_review_markdown_reads_verdicts_only() -> None:
    """只收 `通过`/`不通过`/`存疑`；自由文本（如「差不多」）视为未填，不污染统计。"""
    sheet = "\n".join(
        [
            "| 题号 | 分类 | 人工结论 | 备注 |",
            "| --- | --- | --- | --- |",
            "| fact-01 | simple_fact | 通过 |  |",
            "| fact-02 | simple_fact | 不通过 | 答非所问 |",
            "| fact-03 | simple_fact | 差不多 |  |",
            "| fact-04 | simple_fact | 存疑 |  |",
        ]
    )

    assert parse_review_markdown(sheet) == {
        "fact-01": "通过",
        "fact-02": "不通过",
        "fact-04": "存疑",
    }


def test_parse_review_markdown_tolerates_garbage() -> None:
    assert parse_review_markdown("") == {}
    assert parse_review_markdown("随便一段话\n没有表格") == {}


def test_summarize_review_counts_and_rate() -> None:
    summary = summarize_review({"a": "通过", "b": "通过", "c": "不通过", "d": "存疑"})

    assert summary == {"filled": 4, "pass": 2, "fail": 1, "unsure": 1, "pass_rate": 0.5}
    assert summarize_review({})["pass_rate"] is None  # 未填写时不能报 0%
    assert summarize_review({"x": "乱填"})["filled"] == 0
