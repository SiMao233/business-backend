"""重索引文档（数据修复工具）：按**真实生产路径**重新解析 → 切分 → 向量化 → 替换索引。

用途：
- 把「确定性 ID（`chunk_point_id`）之前索引」的存量文档归一化（让 `chunk.id == vector_id`）；
- 修复内容或向量缺失的文档（配合 `scripts/rag_index_audit.py` 的「缺向量」清单）。

行为与安全：
- 默认 **dry-run**：只列出将被重索引的文档，不动任何数据；加 `--yes` 才真正执行；
- 执行时调用 `process_document_task()` —— ARQ worker 用的**同一个函数**，行为与线上一致
  （含 CAS 抢占、失败标记 failed、确定性 ID 幂等）；
- ⚠️ 会用**当前切分器**重算：切块数量与内容可能与旧版不同（这是预期的，通常更好）；
- ⚠️ 会改变 `chunk.id` → 历史消息 `sources` 快照里的 `chunk_id` 会失效
  （该字段仅用于展示，来源卡片的文档预览/下载靠 document_id/file_id，不受影响）。

用法（**必须在项目根目录执行**：`.env` 与 `storage/` 都是相对路径）：
    .venv/bin/python scripts/rag_reindex.py --stale-ids          # 列出旧方案文档（dry-run）
    .venv/bin/python scripts/rag_reindex.py --stale-ids --yes    # 真正执行
    .venv/bin/python scripts/rag_reindex.py --doc <id> --doc <id> --yes
    .venv/bin/python scripts/rag_reindex.py --missing-vectors --yes   # 修复缺向量的文档
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

from app.core.database import AsyncSessionLocal  # noqa: E402


def norm(value: object) -> str:
    return str(value or "").replace("-", "").lower()


async def stale_document_ids() -> list[tuple[str, str, int]]:
    """旧方案文档：存在 chunk.id != vector_id 的切块。返回 (document_id, name, 切块数)。"""
    sql = (
        "SELECT c.document_id, d.name, COUNT(*) "
        "FROM sys_knowledge_chunk c "
        "LEFT JOIN sys_knowledge_document d ON d.id = c.document_id "
        "WHERE c.vector_id IS NOT NULL AND LOWER(REPLACE(c.id, '-', '')) "
        "      <> LOWER(REPLACE(c.vector_id, '-', '')) "
        "GROUP BY c.document_id, d.name ORDER BY d.name"
    )
    async with AsyncSessionLocal() as db:
        return [(norm(r[0]), r[1] or "", int(r[2])) for r in (await db.execute(text(sql))).all()]


async def missing_vector_document_ids() -> list[tuple[str, str, int]]:
    """缺向量文档：切块的 vector_id 不在 Qdrant 中（需先跑对账脚本确认）。"""
    from app.core.config import get_settings
    from app.modules.ai.vector import get_qdrant_client

    settings = get_settings()
    client = get_qdrant_client()
    point_ids: set[str] = set()
    offset = None
    while True:
        batch, offset = await client.scroll(
            collection_name=settings.qdrant_collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        point_ids.update(norm((p.payload or {}).get("chunk_id")) for p in batch)
        if offset is None:
            break

    sql = (
        "SELECT c.document_id, d.name, COUNT(*) "
        "FROM sys_knowledge_chunk c "
        "LEFT JOIN sys_knowledge_document d ON d.id = c.document_id "
        "GROUP BY c.document_id, d.name"
    )
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(text(sql))).all()

    # 逐文档统计「有多少切块的 vector_id 不在 Qdrant」，只保留存在缺失的文档
    sql_detail = "SELECT document_id, vector_id FROM sys_knowledge_chunk"
    async with AsyncSessionLocal() as db:
        details = (await db.execute(text(sql_detail))).all()
    bad_docs: dict[str, int] = {}
    for doc_id, vector_id in details:
        if norm(vector_id) not in point_ids:
            bad_docs[norm(doc_id)] = bad_docs.get(norm(doc_id), 0) + 1
    return [(d, name, bad_docs[d]) for d, name, _total in rows if d in bad_docs]


async def main() -> int:
    parser = argparse.ArgumentParser(description="重索引文档（默认 dry-run）")
    parser.add_argument("--doc", action="append", default=[], help="文档 ID，可重复")
    parser.add_argument("--stale-ids", action="store_true", help="选中所有旧方案文档")
    parser.add_argument("--missing-vectors", action="store_true", help="选中所有缺向量的文档")
    parser.add_argument("--yes", action="store_true", help="真正执行（默认只列出）")
    args = parser.parse_args()

    targets: list[tuple[str, str, int]] = []
    if args.stale_ids:
        targets += await stale_document_ids()
    if args.missing_vectors:
        targets += await missing_vector_document_ids()
    for doc_id in args.doc:
        targets.append((norm(doc_id), "（手动指定）", 0))

    # 去重（同一个文档可能同时被多个条件选中）
    unique: dict[str, tuple[str, str, int]] = {}
    for doc_id, name, count in targets:
        unique.setdefault(doc_id, (doc_id, name, count))
    targets = list(unique.values())

    if not targets:
        print("没有需要重索引的文档 ✅")
        return 0

    print(f"待重索引 {len(targets)} 个文档：")
    for doc_id, name, count in targets:
        print(f"  {doc_id} {name!r} 涉及切块={count or '-'}")

    if not args.yes:
        print("\n（dry-run：未执行。确认无误后加 --yes）")
        return 0

    from app.modules.knowledge.service import process_document_task

    print()
    failed = 0
    for doc_id, name, _count in targets:
        result = await process_document_task(doc_id)
        ok = bool(result.get("ok"))
        failed += 0 if ok else 1
        print(f"  {'✅' if ok else '❌'} {doc_id} {name!r} → {result}")

    print(f"\n完成：成功 {len(targets) - failed} / 失败 {failed}")
    print("建议接着跑：.venv/bin/python scripts/rag_index_audit.py")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
