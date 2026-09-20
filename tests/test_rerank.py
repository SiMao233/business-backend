"""Reranker 测试：纯函数 / 可替换组件 / 配置降级 / 调用失败降级 / 与检索链路的编排。

不依赖真实 MySQL / Qdrant / 外部 rerank API：
- 纯函数（apply_scores / _extract_results）直接断言；
- openai 客户端用替身替换（捕获构造参数与请求体）；
- 模型实例解析用 monkeypatch 替换 Repository 方法。
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core.config import get_settings
from app.modules.ai import retrievers as retrievers_module
from app.modules.ai.rerankers import (
    MODEL,
    NOOP,
    NoopReranker,
    OpenAICompatibleReranker,
    _extract_results,
    apply_rerank,
    apply_scores,
    build_reranker,
)


def _set(**overrides) -> None:
    """覆盖全局 rag_* 配置（get_settings 为 lru_cache，patch 的是同一实例）。"""
    settings = get_settings()
    for key, value in overrides.items():
        setattr(settings, key, value)


def _hit(content: str = "正文") -> dict:
    """构造一个检索 hit（形状与 retrievers.py 的产出一致）。"""
    return {
        "chunk_id": uuid4().hex,
        "document_id": uuid4().hex,
        "knowledge_base_id": uuid4().hex,
        "seq_no": 1,
        "content": content,
        "score": 0.5,
        "retrievers": ["vector"],
    }


def _install_reranker_client(monkeypatch: pytest.MonkeyPatch, data=None, exc=None) -> dict:
    """替换 openai.AsyncOpenAI，捕获构造参数与 post 请求体。

    `rerankers.py` 在 `__init__` 里做 `from openai import AsyncOpenAI`（延迟导入），
    因此 patch `openai.AsyncOpenAI` 即可生效。
    """
    import openai

    captured: dict = {"init": None, "calls": []}

    class _FakeClient:
        def __init__(self, **kwargs) -> None:
            captured["init"] = kwargs

        async def post(self, path, *, cast_to, body):
            captured["calls"].append({"path": path, "body": body, "cast_to": cast_to})
            if exc is not None:
                raise exc
            return data

    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeClient)
    return captured


def _install_model_lookup(monkeypatch: pytest.MonkeyPatch, provider=None, instance=None) -> None:
    """替换供应商 / 模型实例查询（rerankers.build_reranker 的解析依赖）。"""
    from app.modules.agent.model import repository as model_repo

    async def fake_provider_by_code(self, code):
        return provider

    async def fake_instance_by_provider_and_code(self, provider_id, code):
        return instance

    monkeypatch.setattr(model_repo.ModelProviderRepository, "get_by_code", fake_provider_by_code)
    monkeypatch.setattr(
        model_repo.ModelInstanceRepository,
        "get_by_provider_and_code",
        fake_instance_by_provider_and_code,
    )


def _rerank_provider() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        code="qwen-rerank",
        status=1,
        base_url="https://x/compatible-api/v1",
        api_key="sk-x",
    )


def _rerank_instance(model_type: str = "rerank") -> SimpleNamespace:
    return SimpleNamespace(code="qwen3-rerank", status=1, model_type=model_type)


def _enable_rerank() -> None:
    """把配置切到「rerank 启用且可解析」的完整状态。"""
    _set(
        rag_rerank_enabled=True,
        rerank_provider_code="qwen-rerank",
        rerank_model_code="qwen3-rerank",
        rag_top_k=2,
        rag_candidate_k=30,
        rag_rerank_score_threshold=0.0,
        rag_rerank_timeout=5.0,
        rag_rerank_instruct="",
    )


# ---- NoopReranker ----


async def test_noop_reranker_is_identity() -> None:
    """恒等：不排序、不截断（截断统一由 apply_rerank 负责，保证关闭态口径一致）。"""
    hits = [_hit("A"), _hit("B")]
    result = await NoopReranker().rerank("问题", hits, top_n=1)
    assert result is hits
    assert NoopReranker.name == NOOP
    assert NoopReranker.optional is False  # 恒等实现不可能失败，不参与 optional 降级


# ---- apply_scores（纯函数） ----


def test_apply_scores_maps_by_index_and_sorts_desc() -> None:
    hits = [_hit("A"), _hit("B"), _hit("C")]
    scored = apply_scores(
        hits,
        [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.4}],
    )
    assert [h["content"] for h in scored] == ["C", "A"]  # index 即 documents 下标，按分数降序
    assert [h["rerank_score"] for h in scored] == [0.9, 0.4]


def test_apply_scores_keeps_vector_score_untouched() -> None:
    """rerank 分数写进 rerank_score，不覆盖 score（sources[].score 契约不变）。"""
    hits = [_hit("A")]
    hits[0]["score"] = 0.31
    scored = apply_scores(hits, [{"index": 0, "relevance_score": 0.99}])

    assert scored[0]["score"] == 0.31  # 向量相似度语义保持
    assert scored[0]["rerank_score"] == 0.99
    assert "rerank_score" not in hits[0]  # 不原地修改调用方的 hit


def test_apply_scores_filters_by_threshold() -> None:
    hits = [_hit("A"), _hit("B")]
    scored = apply_scores(
        hits,
        [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.2}],
        threshold=0.5,
    )
    assert [h["content"] for h in scored] == ["A"]


def test_apply_scores_drops_dirty_items() -> None:
    """脏数据不得打断检索链路：越界 / 重复 / 缺字段 / 类型不符一律丢弃。"""
    hits = [_hit("A"), _hit("B")]
    scored = apply_scores(
        hits,
        [
            {"index": 5, "relevance_score": 0.9},  # 越界
            {"index": -1, "relevance_score": 0.9},  # 负数
            {"index": 0, "relevance_score": 0.9},  # 有效
            {"index": 0, "relevance_score": 0.95},  # 重复 index → 丢弃，保留先到的
            {"index": 1},  # 缺分数
            {"index": 1, "relevance_score": "0.5"},  # 分数非数值
            {"index": True, "relevance_score": 0.9},  # bool 不是合法 index
            "not-a-dict",
            None,
        ],
    )
    assert [(h["content"], h["rerank_score"]) for h in scored] == [("A", 0.9)]


def test_apply_scores_empty_or_none_results() -> None:
    assert apply_scores([_hit()], []) == []
    assert apply_scores([_hit()], None) == []


def test_extract_results_tolerates_both_shapes() -> None:
    """兼容 OpenAI 兼容端点（扁平）与原生 DashScope 端点（嵌套）两种响应形状。"""
    flat = [{"index": 0, "relevance_score": 0.8}]
    assert _extract_results({"results": flat}) == flat
    assert _extract_results({"output": {"results": flat}}) == flat
    assert _extract_results(None) == []
    assert _extract_results("oops") == []
    assert _extract_results({"results": "not-a-list"}) == []
    assert _extract_results({"output": {}}) == []


# ---- OpenAICompatibleReranker ----


async def test_reranker_sends_query_and_content_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """请求体：只传 content（不传 embedding_text，避免标题前缀虚高相关性）。"""
    captured = _install_reranker_client(
        monkeypatch, {"results": [{"index": 1, "relevance_score": 0.91}]}
    )
    reranker = OpenAICompatibleReranker(
        model="qwen3-rerank", api_key="sk-x", base_url="https://x/compatible-api/v1", timeout=7.5
    )
    scored = await reranker.rerank("ERR_4012 是什么错误", [_hit("A"), _hit("B")], top_n=30)

    assert captured["init"]["base_url"] == "https://x/compatible-api/v1"
    assert captured["init"]["timeout"] == 7.5
    assert captured["init"]["api_key"] == "sk-x"
    # 必须禁用重试：SDK 默认重试 2 次会把最坏延迟放大到 timeout × 3
    assert captured["init"]["max_retries"] == 0

    call = captured["calls"][0]
    assert call["path"] == "/reranks"
    assert call["body"]["model"] == "qwen3-rerank"
    assert call["body"]["query"] == "ERR_4012 是什么错误"
    assert call["body"]["documents"] == ["A", "B"]
    assert call["body"]["top_n"] == 2  # 不超过候选数
    assert "instruct" not in call["body"]  # 未配置则不下发，走模型默认策略
    assert [h["content"] for h in scored] == ["B"]


async def test_reranker_includes_instruct_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_reranker_client(monkeypatch, {"results": []})
    reranker = OpenAICompatibleReranker(
        model="m",
        api_key=None,
        base_url="u",
        timeout=1.0,
        instruct="Retrieve semantically similar text.",
    )
    await reranker.rerank("q", [_hit()], top_n=1)
    assert captured["calls"][0]["body"]["instruct"] == "Retrieve semantically similar text."


async def test_reranker_uses_default_api_key_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    """供应商未配 api_key 时不得让客户端构造失败（与 vector.py 同一处理方式）。"""
    captured = _install_reranker_client(monkeypatch, {"results": []})
    reranker = OpenAICompatibleReranker(model="m", api_key=None, base_url="u", timeout=1.0)
    await reranker.rerank("q", [_hit()], top_n=1)
    assert captured["init"]["api_key"] == "not-set"


async def test_reranker_empty_hits_skips_request(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_reranker_client(monkeypatch, {"results": []})
    reranker = OpenAICompatibleReranker(model="m", api_key=None, base_url="u", timeout=1.0)
    assert await reranker.rerank("q", [], top_n=5) == []
    assert captured["calls"] == []  # 空候选不发请求


async def test_reranker_applies_score_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_reranker_client(
        monkeypatch,
        {"results": [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.1}]},
    )
    reranker = OpenAICompatibleReranker(
        model="m", api_key=None, base_url="u", timeout=1.0, score_threshold=0.5
    )
    hits = [_hit("A"), _hit("B")]
    assert [h["content"] for h in await reranker.rerank("q", hits, top_n=5)] == ["A"]


# ---- build_reranker（配置解析与降级） ----


async def test_build_reranker_disabled_returns_noop_without_touching_db() -> None:
    _set(rag_rerank_enabled=False)

    class _BoomSession:
        async def execute(self, *args, **kwargs):  # pragma: no cover - 不应被调用
            raise AssertionError("关闭 Reranker 时不得查库")

    reranker = await build_reranker(_BoomSession())  # type: ignore[arg-type]
    assert isinstance(reranker, NoopReranker)
    assert reranker.name == NOOP


async def test_build_reranker_missing_model_config_returns_noop() -> None:
    _set(rag_rerank_enabled=True, rerank_provider_code="", rerank_model_code="")
    assert isinstance(await build_reranker(db=None), NoopReranker)  # type: ignore[arg-type]


async def test_build_reranker_provider_not_found_returns_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set(rag_rerank_enabled=True, rerank_provider_code="nope", rerank_model_code="m")
    _install_model_lookup(monkeypatch, provider=None)
    assert isinstance(await build_reranker(db=None), NoopReranker)  # type: ignore[arg-type]


async def test_build_reranker_provider_disabled_returns_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set(rag_rerank_enabled=True, rerank_provider_code="qwen-rerank", rerank_model_code="m")
    provider = SimpleNamespace(id=uuid4(), code="qwen-rerank", status=0, base_url="u", api_key="k")
    _install_model_lookup(monkeypatch, provider=provider)
    assert isinstance(await build_reranker(db=None), NoopReranker)  # type: ignore[arg-type]


async def test_build_reranker_instance_not_found_returns_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set(rag_rerank_enabled=True, rerank_provider_code="qwen-rerank", rerank_model_code="m")
    _install_model_lookup(monkeypatch, provider=_rerank_provider(), instance=None)
    assert isinstance(await build_reranker(db=None), NoopReranker)  # type: ignore[arg-type]


async def test_build_reranker_instance_disabled_returns_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set(rag_rerank_enabled=True, rerank_provider_code="qwen-rerank", rerank_model_code="m")
    instance = SimpleNamespace(code="qwen3-rerank", status=0, model_type="rerank")
    _install_model_lookup(monkeypatch, provider=_rerank_provider(), instance=instance)
    assert isinstance(await build_reranker(db=None), NoopReranker)  # type: ignore[arg-type]


async def test_build_reranker_wrong_model_type_returns_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """把 chat / embedding 实例误配为 Reranker 时必须降级（否则报错很难定位）。"""
    _set(rag_rerank_enabled=True, rerank_provider_code="qwen-rerank", rerank_model_code="m")
    _install_model_lookup(
        monkeypatch, provider=_rerank_provider(), instance=_rerank_instance("chat")
    )
    assert isinstance(await build_reranker(db=None), NoopReranker)  # type: ignore[arg-type]


async def test_build_reranker_resolves_client_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_rerank()
    _install_model_lookup(monkeypatch, provider=_rerank_provider(), instance=_rerank_instance())
    _install_reranker_client(monkeypatch, {"results": []})

    reranker = await build_reranker(db=None)  # type: ignore[arg-type]

    assert isinstance(reranker, OpenAICompatibleReranker)
    assert reranker.name == MODEL
    assert reranker.optional is True  # 增强能力：失败必须可降级
    assert reranker.model == "qwen3-rerank"
    assert reranker.score_threshold == 0.0


async def test_build_reranker_never_raises_on_resolve_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """解析过程本身出错（如 DB 抖动）也必须降级，不能穿透到检索链路。"""
    from app.modules.agent.model import repository as model_repo

    _enable_rerank()

    async def boom(self, code):
        raise RuntimeError("MySQL 连接中断")

    monkeypatch.setattr(model_repo.ModelProviderRepository, "get_by_code", boom)

    reranker = await build_reranker(db=None)  # type: ignore[arg-type]
    assert isinstance(reranker, NoopReranker)


# ---- apply_rerank（编排 + 降级 + 截断） ----


async def test_apply_rerank_disabled_does_no_external_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关闭态：零外部调用，且结果与改造前一致（融合直接输出 rag_top_k）。"""
    _set(rag_rerank_enabled=False, rag_top_k=2)
    captured = _install_reranker_client(monkeypatch, {"results": []})
    hits = [_hit(str(i)) for i in range(5)]

    result = await apply_rerank(db=None, query="q", hits=hits)  # type: ignore[arg-type]

    assert [h["content"] for h in result] == ["0", "1"]
    assert captured["init"] is None  # 连客户端都没构造
    assert captured["calls"] == []
    assert all(h.get("rerank_score") is None for h in result)


async def test_apply_rerank_reorders_then_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_rerank()
    _install_model_lookup(monkeypatch, provider=_rerank_provider(), instance=_rerank_instance())
    _install_reranker_client(
        monkeypatch,
        {
            "results": [
                {"index": 2, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.8},
                {"index": 0, "relevance_score": 0.1},
            ]
        },
    )
    hits = [_hit("A"), _hit("B"), _hit("C")]

    result = await apply_rerank(db=None, query="q", hits=hits)  # type: ignore[arg-type]

    assert [h["content"] for h in result] == ["C", "B"]  # 重排后截断到 rag_top_k=2


async def test_apply_rerank_degrades_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """调用失败必须退回融合结果：精排失败不能让对话不可用。"""
    _enable_rerank()
    _install_model_lookup(monkeypatch, provider=_rerank_provider(), instance=_rerank_instance())
    _install_reranker_client(monkeypatch, exc=RuntimeError("网关 503"))
    hits = [_hit(str(i)) for i in range(5)]

    result = await apply_rerank(db=None, query="q", hits=hits)  # type: ignore[arg-type]

    assert [h["content"] for h in result] == ["0", "1"]  # 原始顺序 + 截断
    assert all(h.get("rerank_score") is None for h in result)


async def test_apply_rerank_degrades_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """超时同样降级（AsyncOpenAI 超时抛 APITimeoutError，属 Exception 子类）。"""
    _enable_rerank()
    _install_model_lookup(monkeypatch, provider=_rerank_provider(), instance=_rerank_instance())
    _install_reranker_client(monkeypatch, exc=TimeoutError("timeout"))
    hits = [_hit(str(i)) for i in range(3)]

    result = await apply_rerank(db=None, query="q", hits=hits)  # type: ignore[arg-type]
    assert [h["content"] for h in result] == ["0", "1"]


async def test_apply_rerank_empty_hits() -> None:
    assert await apply_rerank(db=None, query="q", hits=[]) == []  # type: ignore[arg-type]


async def test_apply_rerank_threshold_can_drop_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """阈值过滤后可以为空（诚实表明资料不足，不做保底）。"""
    _enable_rerank()
    _set(rag_rerank_score_threshold=0.99)
    _install_model_lookup(monkeypatch, provider=_rerank_provider(), instance=_rerank_instance())
    _install_reranker_client(monkeypatch, {"results": [{"index": 0, "relevance_score": 0.2}]})

    assert await apply_rerank(db=None, query="q", hits=[_hit("A")]) == []  # type: ignore[arg-type]


# ---- 与 retrieve_hits 的编排 ----


def test_build_retriever_uses_candidate_k_when_rerank_enabled() -> None:
    """召回宁多勿漏：启用 Reranker 时融合放大到候选池。"""
    _set(rag_hybrid_enabled=True, rag_rerank_enabled=True, rag_candidate_k=25, rag_top_k=4)
    retriever = retrievers_module.build_retriever()
    assert retriever.top_k == 25  # type: ignore[attr-defined]


def test_build_retriever_uses_top_k_when_rerank_disabled() -> None:
    """关闭态融合上限仍是 rag_top_k（行为与改造前一致）。"""
    _set(rag_hybrid_enabled=True, rag_rerank_enabled=False, rag_candidate_k=25, rag_top_k=4)
    retriever = retrievers_module.build_retriever()
    assert retriever.top_k == 4  # type: ignore[attr-defined]


async def test_retrieve_hits_reranks_then_enriches_only_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """编排：召回 -> 精排 -> 富化（富化只处理最终条数，IN 查询行数更少）。"""
    from app.modules.ai import retrieval as retrieval_module
    from app.modules.knowledge import repository as knowledge_repo

    kb_id = uuid4()
    doc_id = uuid4()
    _enable_rerank()

    async def fake_list_kbs(self, ids):
        return [
            SimpleNamespace(id=kb_id, status=1, name="KB", embedding_model_id=uuid4()),
        ]

    monkeypatch.setattr(knowledge_repo.KnowledgeBaseRepository, "list_by_ids", fake_list_kbs)

    enriched: dict = {}

    async def fake_list_docs(self, ids):
        enriched["doc_ids"] = list(ids)
        return [SimpleNamespace(id=doc_id, name="doc.md", file_id=None)]

    monkeypatch.setattr(knowledge_repo.DocumentRepository, "list_by_ids", fake_list_docs)

    candidates = [_hit(str(i)) for i in range(5)]
    for hit in candidates:
        hit["document_id"] = doc_id.hex
        hit["knowledge_base_id"] = kb_id.hex

    class _FakeRetriever:
        name = "hybrid"

        async def retrieve(self, db, kbs, query):
            return candidates

    # retrieve_hits 在函数内 `from ... import build_retriever`，故 patch 源模块属性即可生效
    monkeypatch.setattr(retrievers_module, "build_retriever", lambda: _FakeRetriever())
    _install_model_lookup(monkeypatch, provider=_rerank_provider(), instance=_rerank_instance())
    _install_reranker_client(
        monkeypatch,
        {
            "results": [
                {"index": 4, "relevance_score": 0.95},
                {"index": 3, "relevance_score": 0.85},
                {"index": 0, "relevance_score": 0.05},
            ]
        },
    )

    hits = await retrieval_module.retrieve_hits(
        db=None, config={"knowledge_ids": [str(kb_id)]}, query="问题"  # type: ignore[arg-type]
    )

    assert [h["content"] for h in hits] == ["4", "3"]  # 精排顺序 + 截断到 rag_top_k=2
    assert hits[0]["document_name"] == "doc.md"  # 富化仍生效
    assert hits[0]["knowledge_base_name"] == "KB"
    assert len(enriched["doc_ids"]) == 2  # 只回查最终条数，不是候选池 30 条
