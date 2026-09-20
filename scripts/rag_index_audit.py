"""RAG 索引对账：MySQL 切块 ↔ Qdrant 向量点（**只读**，不改任何数据）。

为什么需要它：RAG 检索的两条路从不同数据源取 chunk_id（向量路取 Qdrant
`payload.chunk_id`，关键词路取 MySQL），一旦两边 ID 口径不一致，RRF 就无法合并同一个
切块，表现为 `sources` 出现重复条目、白占 `rag_top_k` 名额。

⚠️ 关联键必须用 `sys_knowledge_chunk.vector_id`（= Qdrant 点 ID），**不是** `chunk.id`：
   确定性 ID（`chunk_point_id`）之前索引的存量切块，`chunk.id` 与点 ID 是两个不同的
   uuid4（靠 `vector_id` 列关联）。用 `chunk.id` 对账会把**正常数据误报**成
   「孤儿向量 / 缺向量」，得出方向完全相反的结论。

输出三类结论：
1. 真孤儿：Qdrant 有点、MySQL 无对应切块（多为文档删除后向量未清理）→ 建议清理；
2. 缺向量：MySQL 有切块、Qdrant 无对应点 → 该切块检索不到，建议重索引该文档；
3. 旧方案切块：`chunk.id != vector_id`（仅提示，不影响对账结果）→ 可选归一化清理。

用法（**必须在项目根目录执行**：`.env` 与 `storage/` 都是相对路径）：
    .venv/bin/python scripts/rag_index_audit.py             # 全部知识库
    .venv/bin/python scripts/rag_index_audit.py --kb <id>   # 单个知识库

退出码：0 = 一致；1 = 存在真孤儿或缺失向量（便于接入巡检 / CI）。
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # .env / storage 均为相对路径，统一在项目根目录运行

from sqlalchemy import text  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.database import AsyncSessionLocal  # noqa: E402
from app.modules.ai.vector import get_qdrant_client  # noqa: E402


def norm(value: object) -> str:
    """统一成无连字符小写 hex（Qdrant payload 与 MySQL 两侧的既有形态）。"""
    return str(value or "").replace("-", "").lower()


async def load_chunks(kb_id: str | None) -> list[tuple]:
    """读取切块：(id, document_id, knowledge_base_id, seq_no, vector_id, document_name)。"""
    sql = (
        "SELECT c.id, c.document_id, c.knowledge_base_id, c.seq_no, c.vector_id, d.name "
        "FROM sys_knowledge_chunk c "
        "LEFT JOIN sys_knowledge_document d ON d.id = c.document_id "
    )
    params: dict = {}
    if kb_id:
        sql += "WHERE c.knowledge_base_id = :kb "
        params["kb"] = kb_id
    sql += "ORDER BY c.knowledge_base_id, c.document_id, c.seq_no"
    async with AsyncSessionLocal() as db:
        return list((await db.execute(text(sql), params)).all())


async def load_points(kb_id: str | None) -> list[tuple]:
    """读取 Qdrant 点：(point_id, payload.chunk_id, document_id, seq_no)。"""
    settings = get_settings()
    client = get_qdrant_client()
    out: list[tuple] = []
    offset = None
    while True:
        batch, offset = await client.scroll(
            collection_name=settings.qdrant_collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            payload = point.payload or {}
            if kb_id and payload.get("knowledge_base_id") != kb_id:
                continue
            out.append(
                (
                    str(point.id),
                    payload.get("chunk_id"),
                    payload.get("document_id"),
                    payload.get("seq_no"),
                )
            )
        if offset is None:
            break
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 索引对账（只读）")
    parser.add_argument("--kb", default=None, help="知识库 ID（hex 或带连字符）；不传则全部")
    args = parser.parse_args()
    kb_id = norm(args.kb) if args.kb else None

    chunks = await load_chunks(kb_id)
    points = await load_points(kb_id)
    settings = get_settings()

    chunk_by_vector = {norm(c[4]): c for c in chunks}
    point_by_chunk = {norm(p[1]): p for p in points}

    print(f"collection={settings.qdrant_collection} kb={kb_id or '（全部）'}")
    print(f"MySQL 切块={len(chunks)}  Qdrant 点数={len(points)}")

    orphans = [p for key, p in point_by_chunk.items() if key not in chunk_by_vector]
    missing = [c for key, c in chunk_by_vector.items() if key not in point_by_chunk]
    stale = [c for c in chunks if norm(c[0]) != norm(c[4])]

    print(f"\n真孤儿（Qdrant 有、MySQL 无）={len(orphans)}")
    for point_id, payload_chunk, doc_id, seq_no in orphans[:20]:
        print(f"  point={point_id} chunk={norm(payload_chunk)[:12]} doc={norm(doc_id)[:12]} seq={seq_no}")
    if len(orphans) > 20:
        print(f"  …（其余 {len(orphans) - 20} 条省略）")

    print(f"\n缺向量（MySQL 有、Qdrant 无）={len(missing)}")
    for chunk_id, doc_id, _kb, seq_no, vector_id, doc_name in missing[:20]:
        print(f"  chunk={norm(chunk_id)[:12]} vector_id={norm(vector_id)[:12]} "
              f"doc={norm(doc_id)[:12]} {doc_name!r} seq={seq_no}")
    if len(missing) > 20:
        print(f"  …（其余 {len(missing) - 20} 条省略）")

    stale_docs = sorted({norm(c[1]) for c in stale})
    print(f"\n旧方案切块（chunk.id != vector_id，不影响对账）={len(stale)}"
          f"，涉及 {len(stale_docs)} 个文档")
    if stale_docs:
        print("  → 如需归一化（让 chunk.id == vector_id），见 scripts/rag_reindex.py")

    ok = not orphans and not missing
    print("\n" + ("✅ 对账一致（无真孤儿 / 无缺向量）" if ok else "❌ 对账不一致，见上方明细"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
