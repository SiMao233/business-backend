"""Hybrid Search 测试：关键词抽取 / LIKE 转义 / RRF 融合 / 三路检索器 / 知识库隔离 / 降级。

不依赖真实 MySQL / Qdrant / ARQ：纯函数直接断言，检索器用 monkeypatch 与替身会话验证
「知识库边界是否正确下传」「融合是否正确去重」。
"""

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

import app.modules.ai.vector as vector_module
from app.core.config import get_settings
from app.modules.ai.retrievers import (
    HYBRID,
    KEYWORD,
    VECTOR,
    HybridRetriever,
    KeywordRetriever,
    VectorRetriever,
    build_retriever,
    extract_keywords,
    fusion_key,
    rrf_fuse,
)
from app.modules.knowledge.repository import ChunkRepository, like_pattern


def _set(**overrides) -> None:
    """覆盖全局 rag_* 配置（get_settings 为 lru_cache，patch 的是同一实例）。"""
    settings = get_settings()
    for key, value in overrides.items():
        setattr(settings, key, value)


def _kb(kb_id: UUID | None = None, model_id: UUID | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=kb_id or uuid4(), embedding_model_id=model_id or uuid4())


def _chunk(content: str = "正文", kb_id: UUID | None = None) -> SimpleNamespace:
    chunk_id = uuid4()
    return SimpleNamespace(
        id=chunk_id,
        # 新方案下 vector_id 与 id 相同；旧方案下不同（融合键必须取 vector_id）
        vector_id=chunk_id.hex,
        document_id=uuid4(),
        knowledge_base_id=kb_id or uuid4(),
        seq_no=1,
        content=content,
    )


# ---- extract_keywords（纯函数） ----


def test_extract_keywords_error_code_and_cjk() -> None:
    terms = extract_keywords("ERR_4012 是什么错误")
    assert terms[0] == "ERR_4012"  # 标识符优先级最高
    # 标识符内的子串不再单独产出（否则 LIKE '%ERR%' 会误匹配 error）
    assert "ERR" not in terms
    assert "4012" not in terms
    assert "错误" in terms  # CJK 二元滑窗


def test_extract_keywords_version_and_api_name() -> None:
    terms = extract_keywords("v1.2.3 版本的 get_user_info 接口怎么调用")
    assert "v1.2.3" in terms
    assert "get_user_info" in terms
    assert "get" not in terms  # 标识符子串不再产出
    assert "user" not in terms


def test_extract_keywords_number_priority_over_cjk() -> None:
    terms = extract_keywords("员工迟到超过30分钟怎么处理")
    assert terms[0] == "30"  # 数字优先于 CJK 滑窗
    assert "员工" in terms
    assert "迟到" in terms


def test_extract_keywords_latin_word_and_hyphen_identifier() -> None:
    terms = extract_keywords("HTTP-500 和 HTTP-404 的区别")
    assert "HTTP-500" in terms
    assert "HTTP-404" in terms
    assert "区别" in terms


def test_extract_keywords_dedup_and_limit() -> None:
    terms = extract_keywords("考勤考勤考勤制度制度", max_terms=3)
    assert len(terms) == 3
    assert len(set(terms)) == 3  # 去重


def test_extract_keywords_empty_and_no_terms() -> None:
    assert extract_keywords("") == []
    assert extract_keywords("？？！") == []  # 无标识符 / 拉丁词 / 数字 / CJK
    assert extract_keywords("考勤", max_terms=0) == []


def test_extract_keywords_cjk_single_char_run_has_no_bigram() -> None:
    assert extract_keywords("好") == []


# ---- like_pattern（纯函数） ----


def test_like_pattern_escapes_wildcards() -> None:
    # `_` 是单字符通配符：不转义会误匹配 ERRX4012 / ERR-4012
    assert like_pattern("ERR_4012") == "%ERR\\_4012%"
    assert like_pattern("100%") == "%100\\%%"
    assert like_pattern("a\\b") == "%a\\\\b%"
    assert like_pattern("普通") == "%普通%"


def test_like_pattern_exact_mode() -> None:
    assert like_pattern("ERR_4012", contains=False) == "ERR\\_4012"


# ---- rrf_fuse（纯函数） ----


def _hit(chunk_id: str, score: float | None, source: str) -> dict:
    return {"chunk_id": chunk_id, "score": score, "retrievers": [source]}


def test_rrf_fuse_merges_duplicates_and_keeps_vector_score() -> None:
    """重复结果融合：两路命中同一 chunk → 只保留一条、分数累加、来源合并。"""
    vector_hits = [_hit("a", 0.9, VECTOR), _hit("b", 0.8, VECTOR)]
    keyword_hits = [_hit("b", None, KEYWORD), _hit("c", None, KEYWORD)]
    fused = rrf_fuse([vector_hits, keyword_hits], [1.0, 1.0], 60, 10)

    assert len(fused) == 3  # 去重：a / b / c
    by_id = {h["chunk_id"]: h for h in fused}
    assert by_id["b"]["retrievers"] == [KEYWORD, VECTOR]  # 双路来源
    assert by_id["b"]["score"] == 0.8  # 保留向量相似度
    assert by_id["c"]["score"] is None  # 纯关键词命中 → null
    # 双路命中的 b 得分最高（1/61 + 1/62 > 1/61）
    assert [h["chunk_id"] for h in fused][0] == "b"


def test_rrf_fuse_weights_change_order() -> None:
    vector_hits = [_hit("a", 0.9, VECTOR)]
    keyword_hits = [_hit("b", None, KEYWORD)]
    # 关键词权重压倒向量
    fused = rrf_fuse([vector_hits, keyword_hits], [0.1, 10.0], 60, 10)
    assert [h["chunk_id"] for h in fused] == ["b", "a"]


def test_rrf_fuse_empty_and_truncation() -> None:
    assert rrf_fuse([[], []], [1.0, 1.0], 60, 10) == []
    assert rrf_fuse([], None, 60, 10) == []
    hits = [_hit(str(i), None, KEYWORD) for i in range(10)]
    assert len(rrf_fuse([hits], [1.0], 60, 3)) == 3


def test_rrf_fuse_skips_hits_without_chunk_id() -> None:
    fused = rrf_fuse([[{"score": 0.9, "retrievers": [VECTOR]}]], [1.0], 60, 10)
    assert fused == []


# ---- VectorRetriever ----


class _FakeQdrant:
    """捕获 query_points 入参的假 Qdrant 客户端。"""

    def __init__(self, points: list) -> None:
        self.points = points
        self.kwargs: dict | None = None

    async def query_points(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(points=self.points)


class _FakeEmbeddings:
    async def aembed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


def _install_vector(monkeypatch: pytest.MonkeyPatch, points: list) -> _FakeQdrant:
    fake = _FakeQdrant(points)

    async def fake_build_embeddings(db, model_id):
        return _FakeEmbeddings()

    monkeypatch.setattr(vector_module, "build_embeddings", fake_build_embeddings)
    monkeypatch.setattr(vector_module, "get_qdrant_client", lambda: fake)
    return fake


async def test_vector_retriever_enforces_kb_boundary_and_fetch_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """向量路必须把知识库 hex 列表作为 Qdrant 过滤条件，并使用过取条数。"""
    _set(rag_vector_fetch_k=7, rag_score_threshold=0.3)
    kb = _kb()
    chunk_id = uuid4().hex
    fake = _install_vector(
        monkeypatch,
        [SimpleNamespace(score=0.9, payload={"chunk_id": chunk_id, "seq_no": 1, "content": "正文"})],
    )

    hits = await VectorRetriever().retrieve(db=None, kbs=[kb], query="问题")  # type: ignore[arg-type]

    assert fake.kwargs is not None
    assert fake.kwargs["limit"] == 7
    condition = fake.kwargs["query_filter"].must[0]
    assert condition.key == "knowledge_base_id"  # 知识库边界未被绕过
    assert condition.match.any == [kb.id.hex]
    assert hits[0]["chunk_id"] == chunk_id
    assert hits[0]["score"] == 0.9
    assert hits[0]["retrievers"] == [VECTOR]


async def test_vector_retriever_empty_kbs_skips_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_vector(monkeypatch, [])
    assert await VectorRetriever().retrieve(db=None, kbs=[], query="q") == []  # type: ignore[arg-type]
    assert fake.kwargs is None


# ---- fusion_key（融合键口径） ----


def test_fusion_key_prefers_vector_id() -> None:
    """旧方案：MySQL 主键与点 ID 不同 → 必须取点 ID（否则两路合不掉）。"""
    mysql_pk, point_id = uuid4(), uuid4()
    old_scheme = SimpleNamespace(id=mysql_pk, vector_id=str(point_id))

    assert fusion_key(old_scheme) == point_id.hex
    assert fusion_key(old_scheme) != mysql_pk.hex
    assert "-" not in fusion_key(old_scheme)  # 统一无连字符 hex（Qdrant payload 形态）


def test_fusion_key_same_for_new_scheme() -> None:
    """新方案：id 与点 ID 相同 → 两种写法结果一致。"""
    chunk_id = uuid4()
    assert fusion_key(SimpleNamespace(id=chunk_id, vector_id=chunk_id.hex)) == chunk_id.hex
    assert fusion_key(SimpleNamespace(id=chunk_id, vector_id=str(chunk_id))) == chunk_id.hex


def test_fusion_key_falls_back_to_chunk_id() -> None:
    """vector_id 缺失（理论上不该出现）也不能报错。"""
    chunk_id = uuid4()
    assert fusion_key(SimpleNamespace(id=chunk_id, vector_id=None)) == chunk_id.hex
    assert fusion_key(SimpleNamespace(id=chunk_id)) == chunk_id.hex


# ---- KeywordRetriever ----


def _install_keyword(monkeypatch: pytest.MonkeyPatch, rows: list) -> dict:
    captured: dict = {}

    async def fake_search(self, knowledge_base_ids, keywords, limit):
        captured["kb_ids"] = knowledge_base_ids
        captured["keywords"] = keywords
        captured["limit"] = limit
        return rows

    monkeypatch.setattr(ChunkRepository, "search_by_keywords", fake_search)
    return captured


async def test_keyword_retriever_passes_kbs_and_extracted_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set(rag_keyword_top_k=5, rag_keyword_max_terms=8)
    kb = _kb()
    chunk = _chunk("错误码 ERR_4012 表示认证失败", kb_id=kb.id)
    captured = _install_keyword(monkeypatch, [(chunk, "手册.md")])

    hits = await KeywordRetriever().retrieve(
        db=None, kbs=[kb], query="ERR_4012 是什么错误"  # type: ignore[arg-type]
    )

    assert captured["kb_ids"] == [kb.id]  # 知识库边界下传
    assert captured["limit"] == 5
    assert "ERR_4012" in captured["keywords"]
    assert hits[0]["chunk_id"] == chunk.vector_id  # 与 Qdrant payload.chunk_id 同值（融合键）
    assert hits[0]["knowledge_base_id"] == kb.id.hex
    assert hits[0]["score"] is None  # 保持 sources[].score 语义
    assert hits[0]["retrievers"] == [KEYWORD]


async def test_keyword_retriever_uses_point_id_for_old_scheme_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧方案存量切块：chunk.id ≠ vector_id 时必须返回点 ID，否则与向量路合不掉。"""
    kb = _kb()
    mysql_pk, point_id = uuid4(), uuid4()
    chunk = SimpleNamespace(
        id=mysql_pk,
        vector_id=str(point_id),
        document_id=uuid4(),
        knowledge_base_id=kb.id,
        seq_no=1,
        content="考勤制度 迟到",
    )
    _install_keyword(monkeypatch, [(chunk, "员工手册.txt")])

    hits = await KeywordRetriever().retrieve(
        db=None, kbs=[kb], query="考勤 迟到"  # type: ignore[arg-type]
    )

    assert hits[0]["chunk_id"] == point_id.hex
    assert hits[0]["chunk_id"] != mysql_pk.hex


async def test_keyword_retriever_no_terms_skips_db(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_keyword(monkeypatch, [])
    assert await KeywordRetriever().retrieve(db=None, kbs=[_kb()], query="？？") == []  # type: ignore[arg-type]
    assert captured == {}


async def test_keyword_retriever_empty_kbs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_keyword(monkeypatch, [])
    assert await KeywordRetriever().retrieve(db=None, kbs=[], query="ERR_4012") == []  # type: ignore[arg-type]
    assert captured == {}


async def test_keyword_search_sql_enforces_kb_and_escapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """真实 SQL 形状：知识库 IN 边界 + LIKE 转义 + 命中数排序。"""

    class _Result:
        def all(self) -> list:
            return []

    class _Session:
        def __init__(self) -> None:
            self.statement = None

        async def execute(self, statement):
            self.statement = statement
            return _Result()

    db = _Session()
    kb_ids = [uuid4(), uuid4()]
    await ChunkRepository(db).search_by_keywords(kb_ids, ["ERR_4012", "考勤"], 5)  # type: ignore[arg-type]

    compiled = db.statement.compile()  # type: ignore[union-attr]
    sql = str(compiled)
    assert "sys_knowledge_chunk.knowledge_base_id IN" in sql
    assert "ESCAPE" in sql
    assert "char_length" in sql.lower()
    params = list(compiled.params.values())
    # IN 子句的参数是 list，需要展平后再断言
    flat: list = []
    for value in params:
        if isinstance(value, list | tuple | set):
            flat.extend(value)
        else:
            flat.append(value)
    assert "%ERR\\_4012%" in flat  # 转义后的模式串确实进入绑定参数
    assert set(kb_ids) <= set(flat)


# ---- HybridRetriever ----


async def test_hybrid_fuses_and_dedups(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk_id = uuid4().hex
    _install_vector(
        monkeypatch,
        [
            SimpleNamespace(score=0.9, payload={"chunk_id": chunk_id, "seq_no": 1, "content": "A"}),
            SimpleNamespace(score=0.7, payload={"chunk_id": uuid4().hex, "seq_no": 2, "content": "B"}),
        ],
    )
    kb = _kb()
    keyword_chunk = SimpleNamespace(
        id=UUID(chunk_id),  # 新方案：id 与 vector_id 相同
        vector_id=chunk_id,
        document_id=uuid4(),
        knowledge_base_id=kb.id,
        seq_no=1,
        content="A",
    )
    _install_keyword(monkeypatch, [(keyword_chunk, "doc.md")])

    retriever = HybridRetriever(
        [VectorRetriever(fetch_k=12), KeywordRetriever(top_k=8, max_terms=8)],
        weights=[1.0, 1.0],
        top_k=10,
        rrf_k=60,
    )
    hits = await retriever.retrieve(db=None, kbs=[kb], query="ERR_4012")  # type: ignore[arg-type]

    assert len(hits) == 2  # 同 chunk 被两路命中 → 只保留一条
    assert hits[0]["chunk_id"] == chunk_id  # 双路命中排最前
    assert hits[0]["retrievers"] == [KEYWORD, VECTOR]
    assert hits[0]["score"] == 0.9


async def test_hybrid_merges_old_scheme_chunk_from_both_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """回归：旧方案切块被两路命中时必须合并成一条（修复前会重复成两条）。

    修复前关键词路用 `chunk.id.hex` 当融合键，而旧方案下它 ≠ Qdrant 点 ID，
    于是同一个切块在两路里 key 不同 → `sources` 出现内容相同、分数相同的重复条目。
    """
    point_id = uuid4().hex  # Qdrant 点 ID（= payload.chunk_id）
    mysql_pk = uuid4()  # MySQL 主键，旧方案下与点 ID 是两个不同的 uuid4
    kb = _kb()
    _install_vector(
        monkeypatch,
        [
            SimpleNamespace(
                score=0.9, payload={"chunk_id": point_id, "seq_no": 1, "content": "考勤"}
            )
        ],
    )
    _install_keyword(
        monkeypatch,
        [
            (
                SimpleNamespace(
                    id=mysql_pk,
                    vector_id=point_id,
                    document_id=uuid4(),
                    knowledge_base_id=kb.id,
                    seq_no=1,
                    content="考勤",
                ),
                "员工手册.txt",
            )
        ],
    )

    retriever = HybridRetriever(
        [VectorRetriever(fetch_k=12), KeywordRetriever(top_k=8, max_terms=8)],
        weights=[1.0, 1.0],
        top_k=10,
        rrf_k=60,
    )
    hits = await retriever.retrieve(db=None, kbs=[kb], query="考勤")  # type: ignore[arg-type]

    assert len(hits) == 1  # 修复前 = 2（同一内容重复两条）
    assert hits[0]["retrievers"] == [KEYWORD, VECTOR]  # 两路都命中
    assert hits[0]["chunk_id"] == point_id
    assert hits[0]["score"] == 0.9  # 保留向量相似度语义


async def test_hybrid_truncates_to_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    _set(rag_top_k=2)
    _install_vector(
        monkeypatch,
        [
            SimpleNamespace(
                score=0.9, payload={"chunk_id": uuid4().hex, "seq_no": i, "content": f"C{i}"}
            )
            for i in range(1, 6)
        ],
    )
    _install_keyword(monkeypatch, [])
    retriever = HybridRetriever([VectorRetriever(), KeywordRetriever()], top_k=2)
    hits = await retriever.retrieve(db=None, kbs=[_kb()], query="问题")  # type: ignore[arg-type]
    assert len(hits) == 2


async def test_hybrid_degrades_when_optional_retriever_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关键词路（optional）失败必须降级：向量结果照常返回，不阻断对话。"""
    _install_vector(
        monkeypatch,
        [SimpleNamespace(score=0.9, payload={"chunk_id": uuid4().hex, "seq_no": 1, "content": "A"})],
    )

    class _BoomKeyword:
        name = KEYWORD
        optional = True

        async def retrieve(self, db, kbs, query):
            raise RuntimeError("MySQL 不可用")

    hits = await HybridRetriever([VectorRetriever(), _BoomKeyword()], top_k=10).retrieve(
        db=None, kbs=[_kb()], query="问题"  # type: ignore[arg-type]
    )
    assert len(hits) == 1
    assert hits[0]["retrievers"] == [VECTOR]


async def test_hybrid_reraises_when_primary_retriever_fails() -> None:
    """向量路（非 optional）失败保持既有语义：向上抛，不静默降级。"""

    class _BoomVector:
        name = VECTOR
        optional = False

        async def retrieve(self, db, kbs, query):
            raise RuntimeError("Qdrant 不可用")

    with pytest.raises(RuntimeError):
        await HybridRetriever([_BoomVector()], top_k=10).retrieve(
            db=None, kbs=[_kb()], query="问题"  # type: ignore[arg-type]
        )


async def test_hybrid_all_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_vector(monkeypatch, [])
    _install_keyword(monkeypatch, [])
    retriever = HybridRetriever([VectorRetriever(), KeywordRetriever()], top_k=10)
    assert await retriever.retrieve(db=None, kbs=[_kb()], query="？？") == []  # type: ignore[arg-type]


# ---- build_retriever（配置开关） ----


def test_build_retriever_returns_hybrid_by_default() -> None:
    _set(rag_hybrid_enabled=True)
    retriever = build_retriever()
    assert retriever.name == HYBRID
    assert [r.name for r in retriever.retrievers] == [VECTOR, KEYWORD]  # type: ignore[attr-defined]


def test_build_retriever_can_disable_hybrid() -> None:
    _set(rag_hybrid_enabled=False)
    retriever = build_retriever()
    assert retriever.name == VECTOR
