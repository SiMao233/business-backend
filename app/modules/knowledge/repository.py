"""Knowledge Repository 层：数据访问（CRUD / 查询）。"""

from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageParams, PageResult, paginate
from app.common.repository import BaseRepository
from app.modules.knowledge.model import KnowledgeBase, KnowledgeChunk, KnowledgeDocument


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
        stmt = (
            select(KnowledgeDocument)
            .where(KnowledgeDocument.knowledge_base_id == knowledge_base_id)
            .order_by(KnowledgeDocument.id.desc())
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
            .where(KnowledgeChunk.content.like(f"%{keyword}%"))
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

    # 批量新增切分块（向量化完成后双写落库）
    async def create_many(self, chunks: list[KnowledgeChunk]) -> None:
        self.db.add_all(chunks)
        await self.db.commit()

    # 删除某文档的全部切分块（重新索引 / 删除文档时用）
    async def delete_by_document(self, document_id: UUID) -> None:
        stmt = select(KnowledgeChunk.id).where(KnowledgeChunk.document_id == document_id)
        ids = list((await self.db.execute(stmt)).scalars().all())
        if ids:
            await self.db.execute(
                KnowledgeChunk.__table__.delete().where(KnowledgeChunk.id.in_(ids))
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

