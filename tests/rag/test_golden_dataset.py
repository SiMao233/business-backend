"""Golden Dataset 的离线校验：数据集必须始终"可用且可维护"。

**不需要外部环境**（不连 MySQL / Qdrant / 模型网关）：核对的是数据集自身的结构与约定。
这是一道常驻关卡 —— 任何一次加题/改题如果留下了结构性隐患（重复 id、跨 chunk 题只有一条证据、
证据子串过短导致误命中、分类失衡…），这里立刻失败。

⚠️ 本文件**不校验证据子串是否真的存在于知识库**（那需要连库）：那条检查在
`tests/rag/test_retrieval_eval.py` 里随检索评测一起跑（证据命中率会把失效的金标暴露出来），
另外 `scripts/rag_eval.py --check-dataset` 可单独做一次「数据集 ↔ 索引」对账。
"""

from collections import Counter

import pytest

from app.modules.ai.evaluation import (
    DEFAULT_DATASET_PATH,
    Category,
    ExpectedBehavior,
    GoldenFile,
    known_threshold_keys,
    load_golden_file,
)

# 每类题目至少要有几道：低于这个数说明该维度实际上没被评测覆盖
MIN_PER_CATEGORY = 3
# 证据子串最短长度：太短的片段（如 "null"）几乎必然被任意内容命中，等于没有金标
MIN_EVIDENCE_CHARS = 8


@pytest.fixture(scope="module")
def golden() -> GoldenFile:
    return load_golden_file(DEFAULT_DATASET_PATH)


def test_dataset_file_exists() -> None:
    assert DEFAULT_DATASET_PATH.is_file(), f"Golden Dataset 缺失: {DEFAULT_DATASET_PATH}"


def test_question_count_within_agreed_range(golden: GoldenFile) -> None:
    """题量约定 20~50：太少没有统计意义，太多单人维护不动。"""
    count = len(golden.questions)

    assert 20 <= count <= 50, f"题目数量 {count} 超出约定区间 20~50"


def test_every_category_is_covered(golden: GoldenFile) -> None:
    counts = Counter(question.category for question in golden.questions)

    assert set(counts) == set(Category), f"分类缺失: {set(Category) - set(counts)}"
    for category, count in counts.items():
        assert count >= MIN_PER_CATEGORY, f"分类 {category} 只有 {count} 道题（至少 {MIN_PER_CATEGORY}）"


def test_question_ids_are_unique(golden: GoldenFile) -> None:
    ids = [question.id for question in golden.questions]

    assert len(ids) == len(set(ids))


def test_non_no_answer_questions_are_fully_specified(golden: GoldenFile) -> None:
    """非无答案题必须有：期望答案（或要点）、期望文档、证据子串。"""
    for question in golden.questions:
        if question.is_no_answer:
            continue
        assert question.expected_answer or question.expected_answer_points, question.id
        assert question.expected_source, question.id
        assert question.expected_evidence, question.id


def test_no_answer_questions_expect_abstain_only(golden: GoldenFile) -> None:
    """无答案题不得带证据/来源：它们不参与 Recall，只判「是否拒答」。"""
    for question in golden.questions:
        if question.category != Category.NO_ANSWER.value:
            continue
        assert question.expected_behavior == ExpectedBehavior.ABSTAIN.value, question.id
        assert not question.expected_evidence, question.id
        assert not question.expected_source, question.id


def test_answer_questions_expect_answer_behavior(golden: GoldenFile) -> None:
    for question in golden.questions:
        if question.category == Category.NO_ANSWER.value:
            continue
        assert question.expected_behavior == ExpectedBehavior.ANSWER.value, question.id


def test_evidence_strings_are_distinctive(golden: GoldenFile) -> None:
    """证据子串要够长且互不重复：过短会误命中，重复会让 Recall 重复计分。"""
    for question in golden.questions:
        evidence = question.expected_evidence
        assert len(evidence) == len(set(evidence)), f"{question.id} 存在重复证据"
        for item in evidence:
            assert len(item) >= MIN_EVIDENCE_CHARS, f"{question.id} 证据过短: {item!r}"


def test_cross_chunk_questions_have_multiple_evidence(golden: GoldenFile) -> None:
    """跨 chunk 题的判据就是「证据分散在多处」，只有一条证据说明题目分类错了。"""
    for question in golden.questions:
        if question.category != Category.CROSS_CHUNK.value:
            continue
        assert len(question.expected_evidence) >= 2, f"{question.id} 跨 chunk 题证据不足两条"


def test_multi_turn_and_anaphora_questions_have_history(golden: GoldenFile) -> None:
    """多轮/指代题没有历史就退化成单轮题（也测不出 Query Rewrite）。"""
    for question in golden.questions:
        if question.category not in (Category.MULTI_TURN.value, Category.ANAPHORA.value):
            continue
        assert question.history, question.id
        roles = [role for role, _ in question.history]
        assert roles == ["user", "assistant"], f"{question.id} 历史轮次应为 user/assistant 成对"


def test_anaphora_questions_are_truly_anaphoric(golden: GoldenFile) -> None:
    """指代题的问题里应出现指代词（否则它其实是多轮题，评估意义不同）。"""
    markers = ("这个", "那个", "它", "上述", "刚才")

    for question in golden.questions:
        if question.category != Category.ANAPHORA.value:
            continue
        assert any(marker in question.question for marker in markers), question.id


def test_every_question_has_notes_or_expected_answer(golden: GoldenFile) -> None:
    """`notes` 记录出题依据；无答案题因为容易引起争议，强制要求写明理由。"""
    for question in golden.questions:
        if question.is_no_answer:
            assert question.notes, f"{question.id} 无答案题必须写 notes（说明为何库里没有）"


def test_each_dataset_points_to_a_knowledge_base(golden: GoldenFile) -> None:
    for dataset in golden.datasets:
        assert dataset.questions, f"{dataset.name} 没有题目"
        assert len(dataset.knowledge_base_id) in (32, 36), dataset.knowledge_base_id
        assert dataset.name


def test_thresholds_use_known_metric_names(golden: GoldenFile) -> None:
    """门禁阈值只能针对真实存在的指标名（拼错会让门禁静默失效）。

    数据集里用 snake_case（`hit_at_5`）便于手写，报告里用 `hit@5` 便于阅读，
    映射由 `threshold_to_metric()` 负责 —— 这里校验键是否在允许集合内。
    """
    unknown = set(golden.thresholds) - known_threshold_keys()

    assert not unknown, f"阈值里出现未知指标: {sorted(unknown)}"


def test_threshold_values_are_in_range(golden: GoldenFile) -> None:
    for name, value in golden.thresholds.items():
        assert 0.0 <= value <= 1.0, f"阈值 {name}={value} 不在 [0,1] 区间"
