"""RAG 索引生命周期测试：确定性 ID、payload 组装、分批 embedding、任务编排与失败态。

不依赖真实 MySQL / Qdrant / ARQ：
- 纯函数（chunk_point_id / build_index_payload / embed_in_batches）直接断言；
- process_document_task 的编排顺序用 monkeypatch 替换 repository / FileService /
  vector / parser / AsyncSessionLocal，并记录调用次序。

核心不变式（对应"重新索引一致性"目标）：
1. 清旧（Qdrant + MySQL）先于写新；
2. 同一文档重复索引得到完全相同的 chunk/point ID（幂等）；
3. 只读计算阶段失败时旧索引完好（无任何清理/写入调用）；
4. 任务被取消（CancelledError）时文档仍被标记 failed，可通过 API 恢复。
"""

import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

import app.core.database as database_module
import app.modules.ai.vector as vector_module
import app.modules.knowledge.parser as parser_module
from app.core.exceptions import BizError
from app.modules.knowledge import service as knowledge_service
from app.modules.knowledge.parser import ChunkDraft, ParsedBlock
from app.modules.knowledge.repository import DocumentRepository
from app.modules.knowledge.service import (
    build_index_payload,
    chunk_point_id,
)

# ---- 确定性 ID ----


def test_chunk_point_id_deterministic() -> None:
    doc_id = uuid4()
    assert chunk_point_id(doc_id, 1) == chunk_point_id(doc_id, 1)
    assert chunk_point_id(doc_id, 1) == UUID(str(chunk_point_id(doc_id, 1)))


def test_chunk_point_id_differs_by_document_and_seq() -> None:
    doc_a, doc_b = uuid4(), uuid4()
    assert chunk_point_id(doc_a, 1) != chunk_point_id(doc_b, 1)
    assert chunk_point_id(doc_a, 1) != chunk_point_id(doc_a, 2)


# ---- build_index_payload ----


def test_build_index_payload_shapes_and_hex_payload() -> None:
    doc_id, kb_id = uuid4(), uuid4()
    drafts = [
        ChunkDraft(content="第一块", section_path="手册 > 一、总则"),
        ChunkDraft(content="第二块", page_no=2, chunk_type="table"),
        ChunkDraft(content="第三块"),
    ]
    vectors = [[0.1], [0.2], [0.3]]
    points, records = build_index_payload(doc_id, kb_id, drafts, vectors)
    assert len(points) == len(records) == 3
    for i, (point, record) in enumerate(zip(points, records, strict=True), start=1):
        draft = drafts[i - 1]
        expected = chunk_point_id(doc_id, i)
        assert point["id"] == str(expected)
        assert point["payload"]["chunk_id"] == expected.hex
        assert point["payload"]["document_id"] == doc_id.hex
        assert point["payload"]["knowledge_base_id"] == kb_id.hex
        assert point["payload"]["seq_no"] == i
        assert point["payload"]["content"] == draft.content
        # 结构元数据同时进 payload 与 MySQL
        assert point["payload"]["section_path"] == draft.section_path
        assert point["payload"]["page_no"] == draft.page_no
        assert point["payload"]["chunk_type"] == draft.chunk_type
        assert record.id == expected
        assert record.document_id == doc_id
        assert record.knowledge_base_id == kb_id
        assert record.seq_no == i
        assert record.vector_id == str(expected)
        assert record.char_count == len(draft.content)
        assert record.section_path == draft.section_path
        assert record.page_no == draft.page_no
        assert record.chunk_type == draft.chunk_type


def test_build_index_payload_strict_length_mismatch() -> None:
    # 向量数少于切分数：必须显式报错，而不是静默截断丢数据
    drafts = [ChunkDraft(content="a"), ChunkDraft(content="b")]
    with pytest.raises(BizError):
        build_index_payload(uuid4(), uuid4(), drafts, [[0.1]])


# ---- DocumentRepository.mark_parsing（CAS 抢占判定） ----


class _CapturingSession:
    """捕获 UPDATE 语句并返回指定 rowcount 的会话替身。"""

    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount
        self.statement = None
        self.committed = False

    async def execute(self, statement):
        self.statement = statement
        return SimpleNamespace(rowcount=self.rowcount)

    async def commit(self) -> None:
        self.committed = True


async def test_mark_parsing_rowcount_semantics() -> None:
    # rowcount == 1 → 抢到处理权
    db = _CapturingSession(rowcount=1)
    assert await DocumentRepository(db).mark_parsing(uuid4()) is True
    assert db.committed is True
    # 语句必须是针对 sys_knowledge_document 的 UPDATE
    assert db.statement is not None
    assert "UPDATE sys_knowledge_document" in str(db.statement)

    # rowcount == 0 → 已被其它任务抢占
    assert await DocumentRepository(_CapturingSession(rowcount=0)).mark_parsing(uuid4()) is False


# ---- embed_in_batches ----


class _FakeEmbeddings:
    """记录每批输入的假 embedding 客户端。"""

    def __init__(self, fail: BaseException | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail is not None:
            raise self.fail
        self.calls.append(texts)
        return [[float(len(text)), 0.0] for text in texts]


async def test_embed_in_batches_respects_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "embedding_batch_size", 2)
    fake = _FakeEmbeddings()
    vectors = await knowledge_service.embed_in_batches(fake, ["a", "b", "c", "d", "e"])
    assert fake.calls == [["a", "b"], ["c", "d"], ["e"]]
    assert len(vectors) == 5


async def test_embed_in_batches_guards_zero_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "embedding_batch_size", 0)
    fake = _FakeEmbeddings()
    await knowledge_service.embed_in_batches(fake, ["a", "b"])
    assert fake.calls == [["a"], ["b"]]


# ---- process_document_task：编排与失败态 ----


class _FakeSession:
    """AsyncSessionLocal 替身：仅提供 async 上下文协议。"""

    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeDocRepo:
    def __init__(self, db: object) -> None:
        self.statuses: list[str] = []
        self.doc: SimpleNamespace | None = None
        # CAS 抢占结果：False 模拟"已有任务在处理该文档"
        self.mark_parsing_result = True
        self.mark_parsing_calls = 0

    async def get_by_id(self, document_id: UUID) -> SimpleNamespace | None:
        return self.doc

    async def mark_parsing(self, document_id: UUID) -> bool:
        self.mark_parsing_calls += 1
        return self.mark_parsing_result

    async def update(self, doc: SimpleNamespace) -> SimpleNamespace:
        self.statuses.append(doc.status)
        return doc


class _FakeKBRepo:
    def __init__(self, db: object) -> None:
        self.kb: SimpleNamespace | None = None

    async def get_by_id(self, kb_id: UUID) -> SimpleNamespace | None:
        return self.kb


class _FakeChunkRepo:
    def __init__(self, db: object) -> None:
        self.calls: list[tuple[str, object]] = []

    async def delete_by_document(self, document_id: UUID) -> None:
        self.calls.append(("delete_by_document", document_id))

    async def create_many(self, records: list) -> None:
        self.calls.append(("create_many", records))


class _FakeFileService:
    def __init__(self, db: object) -> None:
        pass

    async def get_content(self, file_id: UUID):
        return SimpleNamespace(data=b"file-bytes", path=None)


def _make_doc() -> tuple[UUID, SimpleNamespace, SimpleNamespace]:
    doc_id = uuid4()
    doc = SimpleNamespace(
        id=doc_id,
        knowledge_base_id=uuid4(),
        name="测试文档.txt",
        file_id=uuid4(),
        status="pending",
        error_message=None,
        chunk_count=0,
    )
    kb = SimpleNamespace(embedding_model_id=uuid4())
    return doc_id, doc, kb


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    doc_repo: _FakeDocRepo,
    kb_repo: _FakeKBRepo,
    chunk_repo: _FakeChunkRepo,
    calls: list,
    embeddings: _FakeEmbeddings,
) -> None:
    """替换任务的全部外部依赖（DB / 文件 / 向量 / 解析）。"""

    async def fake_build_embeddings(db: object, model_id: UUID | None):
        calls.append("build_embeddings")
        return embeddings

    async def fake_ensure_collection(dim: int) -> None:
        calls.append(("ensure_collection", dim))

    async def fake_delete_by_filter(**filters: str) -> None:
        calls.append(("delete_by_filter", dict(filters)))

    async def fake_upsert_chunks(points: list[dict]) -> None:
        calls.append(("upsert_chunks", points))

    monkeypatch.setattr(knowledge_service, "DocumentRepository", lambda db: doc_repo)
    monkeypatch.setattr(knowledge_service, "KnowledgeBaseRepository", lambda db: kb_repo)
    monkeypatch.setattr(knowledge_service, "ChunkRepository", lambda db: chunk_repo)
    monkeypatch.setattr(knowledge_service, "FileService", lambda db: _FakeFileService(db))
    monkeypatch.setattr(vector_module, "build_embeddings", fake_build_embeddings)
    monkeypatch.setattr(vector_module, "ensure_collection", fake_ensure_collection)
    monkeypatch.setattr(vector_module, "delete_by_filter", fake_delete_by_filter)
    monkeypatch.setattr(vector_module, "upsert_chunks", fake_upsert_chunks)
    monkeypatch.setattr(
        parser_module,
        "parse_document",
        lambda data, ext: [ParsedBlock(text="解析后的文本")],
    )
    monkeypatch.setattr(
        parser_module,
        "split_blocks",
        lambda blocks: [
            ChunkDraft(content="第一块内容"),
            ChunkDraft(content="第二块内容"),
            ChunkDraft(content="第三块内容"),
        ],
    )
    # 任务内 `from app.core.database import AsyncSessionLocal` 在调用时才绑定，
    # 因此 patch 模块属性即可生效
    monkeypatch.setattr(database_module, "AsyncSessionLocal", _FakeSession)


def _call_names(calls: list) -> list[str]:
    return [c if isinstance(c, str) else c[0] for c in calls]


async def test_process_document_replaces_index_in_order_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id, doc, kb = _make_doc()
    doc_repo = _FakeDocRepo(object())
    doc_repo.doc = doc
    kb_repo = _FakeKBRepo(object())
    kb_repo.kb = kb
    chunk_repo = _FakeChunkRepo(object())
    calls: list = []
    _install_fakes(monkeypatch, doc_repo, kb_repo, chunk_repo, calls, _FakeEmbeddings())

    result = await knowledge_service.process_document_task(str(doc_id))

    assert result == {"ok": True, "chunk_count": 3}
    # CAS 抢占成功，最终落库只有 parsed（parsing 由 CAS 以 Core UPDATE 写入）
    assert doc_repo.mark_parsing_calls == 1
    assert doc_repo.statuses == ["parsed"]
    assert doc.status == "parsed" and doc.chunk_count == 3

    # 阶段 3 顺序：清旧向量 → 清旧 chunk → 写新向量 → 写新 chunk
    names = _call_names(calls)
    assert names.index("delete_by_filter") < names.index("upsert_chunks")
    chunk_names = [name for name, _ in chunk_repo.calls]
    assert chunk_names.index("delete_by_document") < chunk_names.index("create_many")
    # 清理按 document_id（hex）过滤，与 Qdrant payload 约定一致
    assert ("delete_by_filter", {"document_id": doc_id.hex}) in calls

    # 确定性 ID：重复执行得到完全相同的 point/chunk ID
    first_points = next(c[1] for c in calls if isinstance(c, tuple) and c[0] == "upsert_chunks")
    first_records = next(r for name, r in chunk_repo.calls if name == "create_many")
    await knowledge_service.process_document_task(str(doc_id))
    second_points = next(c[1] for c in calls if isinstance(c, tuple) and c[0] == "upsert_chunks")
    second_records = next(r for name, r in chunk_repo.calls if name == "create_many")
    assert [p["id"] for p in second_points] == [p["id"] for p in first_points]
    assert [r.id for r in second_records] == [r.id for r in first_records]


async def test_process_document_skips_when_cas_not_acquired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CAS 抢不到处理权（已有任务在跑）时必须直接跳过，且不做任何索引写入。

    这取代了之前用 ARQ 固定 job_id 做去重的做法：那种方式会在任务结束后的
    keep_result 窗口（默认 1 小时）内把重试入队静默丢弃，导致文档卡 pending。
    """
    doc_id, doc, kb = _make_doc()
    doc_repo = _FakeDocRepo(object())
    doc_repo.doc = doc
    doc_repo.mark_parsing_result = False
    kb_repo = _FakeKBRepo(object())
    kb_repo.kb = kb
    chunk_repo = _FakeChunkRepo(object())
    calls: list = []
    _install_fakes(monkeypatch, doc_repo, kb_repo, chunk_repo, calls, _FakeEmbeddings())

    result = await knowledge_service.process_document_task(str(doc_id))

    assert result == {"ok": True, "skipped": "already parsing"}
    assert doc_repo.mark_parsing_calls == 1
    # 未标记任何终态，也完全没碰旧索引
    assert doc_repo.statuses == []
    assert calls == []
    assert chunk_repo.calls == []


async def test_process_document_read_phase_failure_keeps_old_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc_id, doc, kb = _make_doc()
    doc_repo = _FakeDocRepo(object())
    doc_repo.doc = doc
    kb_repo = _FakeKBRepo(object())
    kb_repo.kb = kb
    chunk_repo = _FakeChunkRepo(object())
    calls: list = []
    # embedding 失败：属于只读计算阶段，旧索引必须完好
    _install_fakes(
        monkeypatch, doc_repo, kb_repo, chunk_repo, calls,
        _FakeEmbeddings(fail=BizError("embedding 服务不可用")),
    )

    result = await knowledge_service.process_document_task(str(doc_id))

    assert result["ok"] is False
    assert doc.status == "failed"
    # errorMessage 是面向用户的纯业务文案（不带阶段前缀）
    assert doc.error_message == "embedding 服务不可用"
    # 失败阶段只出现在任务返回值 / 日志里
    assert result["stage"] == "embedding"
    # 没有任何清理/写入调用（旧索引未被破坏）
    assert not any(
        isinstance(c, tuple) and c[0] in {"delete_by_filter", "upsert_chunks"} for c in calls
    )
    assert chunk_repo.calls == []


async def test_process_document_cancelled_marks_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ARQ 超时取消（CancelledError）不得让文档永久卡在 parsing。"""
    doc_id, doc, kb = _make_doc()
    doc_repo = _FakeDocRepo(object())
    doc_repo.doc = doc
    kb_repo = _FakeKBRepo(object())
    kb_repo.kb = kb
    chunk_repo = _FakeChunkRepo(object())
    calls: list = []
    _install_fakes(
        monkeypatch, doc_repo, kb_repo, chunk_repo, calls,
        _FakeEmbeddings(fail=asyncio.CancelledError()),
    )

    inner = asyncio.ensure_future(knowledge_service.process_document_task(str(doc_id)))
    with pytest.raises(asyncio.CancelledError):
        await asyncio.shield(inner)

    assert inner.cancelled() is True
    assert doc.status == "failed"
    # CancelledError 的 str() 为空，必须落到兜底文案而非空串
    assert doc.error_message == "任务被中断（可能超时或服务重启），请稍后重试"
    # 取消发生在只读计算阶段，旧索引未被破坏
    assert not any(
        isinstance(c, tuple) and c[0] in {"delete_by_filter", "upsert_chunks"} for c in calls
    )
