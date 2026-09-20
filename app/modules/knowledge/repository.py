"""Knowledge Repository 层：数据访问（CRUD / 查询）。"""

from typing import Any
from uuid import UUID

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import DocumentStatus
from app.common.pagination import PageParams, PageResult, paginate
from app.common.repository import BaseRepository
from app.modules.knowledge.model import KnowledgeBase, KnowledgeChunk, KnowledgeDocument

# LIKE 转义字符（必须与 SQL 里的 ESCAPE 子句保持一致）
LIKE_ESCAPE = "\\"


def like_pattern(term: str, contains: bool = True) -> str:
    """把关键词转成安全的 LIKE 模式串（转义 `\\` `%` `_`）。

    为什么必须转义：`_` 是 LIKE 的单字符通配符，`%` 是任意长度通配符。
    例如 `ERR_4012` 不转义会匹配到 `ERRX4012`、`ERR-4012` 等无关内容 ——
    而错误码 / 版本号 / 常量名恰恰大量使用 `_` 与 `.`。

    `contains=False` 时返回精确匹配模式（无前后通配符）。
    """
    escaped = (
        term.replace(LIKE_ESCAPE, LIKE_ESCAPE * 2)
        .replace("%", f"{LIKE_ESCAPE}%")
        .replace("_", f"{LIKE_ESCAPE}_")
    )
    return f"%{escaped}%" if contains else escaped


class KnowledgeBaseRepository(BaseRepository):
    """知识库数据访问仓库。"""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, KnowledgeBase)

    # 根据主键 ID 查询知识库
    async def get_by_id(self, knowledge_base_id: UUID) -> KnowledgeBase | None:
        stmt = select(KnowledgeBase).where(KnowledgeBase.id == knowledge_base_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 按主键批量查询知识库（RAG 回查知识库名，避免逐条查询）
    async def list_by_ids(self, knowledge_base_ids: list[UUID]) -> list[KnowledgeBase]:
        if not knowledge_base_ids:
            return []
        stmt = select(KnowledgeBase).where(KnowledgeBase.id.in_(knowledge_base_ids))
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # 根据业务编码查询（唯一性校验，可排除自身）
    async def get_by_code(self, code: str, exclude_id: UUID | None = None) -> KnowledgeBase | None:
        stmt = select(KnowledgeBase).where(KnowledgeBase.code == code)
        if exclude_id is not None:
            stmt = stmt.where(KnowledgeBase.id != exclude_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 分页查询知识库列表，支持关键字 / 组织 / 状态筛选
    async def list_page(
        self,
        params: PageParams,
        keyword: str | None = None,
        organization_id: UUID | None = None,
        status: int | None = None,
    ) -> PageResult[Any]:
        stmt = select(KnowledgeBase).order_by(KnowledgeBase.id.desc())
        if keyword:
            like = f"%{keyword}%"
            stmt = stmt.where(KnowledgeBase.name.like(like) | KnowledgeBase.code.like(like))
        if organization_id is not None:
            stmt = stmt.where(KnowledgeBase.organization_id == organization_id)
        if status is not None:
            stmt = stmt.where(KnowledgeBase.status == status)
        return await paginate(self.db, stmt, params)

    # 新增知识库：写入并提交，刷新后返回对象
    async def create(self, kb: KnowledgeBase) -> KnowledgeBase:
        self.db.add(kb)
        await self.db.commit()
        stmt = select(KnowledgeBase).where(KnowledgeBase.id == kb.id)
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 更新知识库：提交变更并刷新，返回更新后的对象
    async def update(self, kb: KnowledgeBase) -> KnowledgeBase:
        await self.db.commit()
        stmt = select(KnowledgeBase).where(KnowledgeBase.id == kb.id)
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 删除知识库：从数据库移除并提交事务（文档/块由 DB CASCADE 删除）
    async def delete(self, kb: KnowledgeBase) -> None:
        await self.db.delete(kb)
        await self.db.commit()


class DocumentRepository(BaseRepository):
    """知识库文档数据访问仓库。"""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, KnowledgeDocument)

    # 根据主键 ID 查询文档
    async def get_by_id(self, document_id: UUID) -> KnowledgeDocument | None:
        stmt = select(KnowledgeDocument).where(KnowledgeDocument.id == document_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 按主键批量查询文档（RAG 来源回查文档名 / 源文件 ID，避免逐条查询）
    async def list_by_ids(self, document_ids: list[UUID]) -> list[KnowledgeDocument]:
        if not document_ids:
            return []
        stmt = select(KnowledgeDocument).where(KnowledgeDocument.id.in_(document_ids))
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # 分页查询某知识库下的文档列表（按创建时间倒序）
    async def list_page(
        self,
        knowledge_base_id: UUID,
        params: PageParams,
        keyword: str | None = None,
        status: str | None = None,
    ) -> PageResult[Any]:
        # 次键 id 不可省：create_time 是 MySQL DATETIME（秒级精度），
        # 同秒上传的文档会取到相同值，仅按 create_time 排序结果不稳定，
        # 配合 LIMIT/OFFSET 分页会出现「同一条出现在两页 / 被跳过」。
        stmt = (
            select(KnowledgeDocument)
            .where(KnowledgeDocument.knowledge_base_id == knowledge_base_id)
            .order_by(KnowledgeDocument.create_time.desc(), KnowledgeDocument.id.desc())
        )
        if keyword:
            like = f"%{keyword}%"
            stmt = stmt.where(KnowledgeDocument.name.like(like))
        if status:
            stmt = stmt.where(KnowledgeDocument.status == status)
        return await paginate(self.db, stmt, params)

    # 新增文档：写入并提交，刷新后返回对象
    async def create(self, doc: KnowledgeDocument) -> KnowledgeDocument:
        self.db.add(doc)
        await self.db.commit()
        stmt = select(KnowledgeDocument).where(KnowledgeDocument.id == doc.id)
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 抢占文档处理权：仅当当前状态不是 parsing 时置为 parsing，返回是否抢占成功
    # 用于后台索引任务的并发保护（CAS，不依赖任务队列的去重语义）：
    # 抢不到说明已有任务在处理该文档，调用方应直接跳过，避免并发重复索引。
    async def mark_parsing(self, document_id: UUID) -> bool:
        result = await self.db.execute(
            update(KnowledgeDocument)
            .where(
                KnowledgeDocument.id == document_id,
                KnowledgeDocument.status != DocumentStatus.PARSING.value,
            )
            .values(status=DocumentStatus.PARSING.value)
            # 关闭 ORM 同步：调用方自行维护内存对象，避免额外 SELECT / 对象过期
            .execution_options(synchronize_session=False)
        )
        await self.db.commit()
        return result.rowcount == 1

    # 更新文档：提交变更并刷新，返回更新后的对象
    async def update(self, doc: KnowledgeDocument) -> KnowledgeDocument:
        await self.db.commit()
        stmt = select(KnowledgeDocument).where(KnowledgeDocument.id == doc.id)
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 删除文档：从数据库移除并提交事务（切分块由 DB CASCADE 删除）
    async def delete(self, doc: KnowledgeDocument) -> None:
        await self.db.delete(doc)
        await self.db.commit()


class ChunkRepository(BaseRepository):
    """文档切分块数据访问仓库（向量化流程使用）。"""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, KnowledgeChunk)

    # 查询某文档的全部切分块（按序号升序）
    async def list_by_document(self, document_id: UUID) -> list[KnowledgeChunk]:
        stmt = (
            select(KnowledgeChunk)
            .where(KnowledgeChunk.document_id == document_id)
            .order_by(KnowledgeChunk.seq_no.asc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # 按关键字搜索某知识库内的切分块（JOIN 文档名，content LIKE，分页）
    async def search_by_keyword(
        self,
        knowledge_base_id: UUID,
        keyword: str,
        params: PageParams,
    ) -> PageResult[Any]:
        stmt = (
            select(KnowledgeChunk, KnowledgeDocument.name)
            .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
            .where(KnowledgeChunk.knowledge_base_id == knowledge_base_id)
            # 转义 LIKE 通配符：否则用户搜索 ERR_4012 会误匹配 ERRX4012
            .where(KnowledgeChunk.content.like(like_pattern(keyword), escape=LIKE_ESCAPE))
            .order_by(KnowledgeChunk.id.desc())
        )
        # 手动分页：paginate 的 scalars() 会丢失 JOIN 的第二列（文档名），需保留元组
        count_stmt = select(func.count()).select_from(stmt.order_by(None).subquery())
        total = (await self.db.execute(count_stmt)).scalar_one()
        rows = await self.db.execute(
            stmt.offset((params.page - 1) * params.pageSize).limit(params.pageSize)
        )
        items = list(rows.all())
        return PageResult(list=items, total=total, page=params.page, pageSize=params.pageSize)

    # 多关键词匹配（Hybrid Search 关键词路）：LIKE 精确子串 + 命中词数排序
    async def search_by_keywords(
        self,
        knowledge_base_ids: list[UUID],
        keywords: list[str],
        limit: int,
    ) -> list[tuple[KnowledgeChunk, str]]:
        """按知识库边界做多关键词 LIKE 匹配，返回 (chunk, 文档名) 列表。

        `knowledge_base_ids` 是**强制检索边界**（Hybrid 的向量路与关键词路的边界必须一致），
        走 `knowledge_base_id IN (...)` 先用索引收窄，再做 LIKE 过滤。

        为什么用 LIKE 而不是 FULLTEXT：
        - 目标场景（错误码 / 版本号 / API 名 / 专有名词）本质是**精确子串**匹配，
          LIKE 与之天然对齐，而 ngram 分词会把标识符切成 bigram 变成近似匹配；
        - InnoDB FULLTEXT 的 relevance 是 TF-IDF 变体（非 BM25），而 RRF 只用排名、
          不需要校准分数，故没有引入更多搜索基础设施的必要。
        ⚠️ 升级触发条件：chunk 总量 > 5 万，或本方法 p95 延迟 > 100ms 时，
        再为 content 增加 `FULLTEXT ... WITH PARSER ngram` 索引（改动范围仅限本方法 + 1 个迁移）。

        排序下推到 SQL：命中关键词个数降序 → 正文长度升序（短块表达更集中）。
        """
        if not knowledge_base_ids or not keywords:
            return []
        conditions = [
            KnowledgeChunk.content.like(like_pattern(term), escape=LIKE_ESCAPE)
            for term in keywords
        ]
        # 命中词数（MySQL 中布尔表达式相加即为计数），用于排序
        match_count = conditions[0]
        for condition in conditions[1:]:
            match_count = match_count + condition
        stmt = (
            select(KnowledgeChunk, KnowledgeDocument.name)
            .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
            .where(KnowledgeChunk.knowledge_base_id.in_(knowledge_base_ids))
            .where(or_(*conditions))
            .order_by(match_count.desc(), func.char_length(KnowledgeChunk.content).asc())
            .limit(max(1, limit))
        )
        rows = await self.db.execute(stmt)
        return list(rows.all())

    # 批量新增切分块（向量化完成后双写落库）
    async def create_many(self, chunks: list[KnowledgeChunk]) -> None:
        self.db.add_all(chunks)
        await self.db.commit()

    # 删除某文档的全部切分块（重新索引 / 删除文档时用）
    async def delete_by_document(self, document_id: UUID) -> None:
        await self.db.execute(
            KnowledgeChunk.__table__.delete().where(KnowledgeChunk.document_id == document_id)
        )
        await self.db.commit()

    # 删除某知识库的全部切分块（删除知识库时用）
    async def delete_by_knowledge_base(self, knowledge_base_id: UUID) -> None:
        stmt = select(KnowledgeChunk.id).where(KnowledgeChunk.knowledge_base_id == knowledge_base_id)
        ids = list((await self.db.execute(stmt)).scalars().all())
        if ids:
            await self.db.execute(
                KnowledgeChunk.__table__.delete().where(KnowledgeChunk.id.in_(ids))
            )
            await self.db.commit()

