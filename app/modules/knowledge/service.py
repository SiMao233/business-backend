"""Knowledge Service 层：业务规则、权限判断、流程编排。

知识库 CRUD + 文档管理（上传/删除/列表）+ 文档向量化后台任务编排。
文档处理流水线：上传 → 入队 ARQ → 解析 → 切分 → embedding → 写 Qdrant → 双写 MySQL chunk。
"""

from uuid import UUID, uuid4

from fastapi import UploadFile
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import DocumentStatus, FileBizType, ModelType
from app.common.pagination import PageParams, PageResult
from app.core.exceptions import BizError, NotFoundError, ValidateError
from app.modules.agent.model.repository import ModelInstanceRepository
from app.modules.file.service import FileService
from app.modules.knowledge.model import KnowledgeBase, KnowledgeChunk, KnowledgeDocument
from app.modules.knowledge.repository import (
    ChunkRepository,
    DocumentRepository,
    KnowledgeBaseRepository,
)
from app.modules.knowledge.schema import (
    ChunkOut,
    ChunkSearchOut,
    ChunkSearchQuery,
    DocumentOut,
    DocumentQuery,
    KnowledgeBaseCreate,
    KnowledgeBaseOut,
    KnowledgeBaseQuery,
    KnowledgeBaseUpdate,
)


class KnowledgeService:
    """知识库管理服务。"""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repo = KnowledgeBaseRepository(db)
        self.doc_repo = DocumentRepository(db)
        self.chunk_repo = ChunkRepository(db)
        self.instance_repo = ModelInstanceRepository(db)

    # 分页查询知识库列表
    async def list_page(self, query: KnowledgeBaseQuery) -> PageResult[KnowledgeBaseOut]:
        page_params = PageParams(page=query.page, pageSize=query.pageSize)
        page = await self.repo.list_page(
            page_params,
            keyword=query.keyword,
            organization_id=query.organization_id,
            status=query.status,
        )
        return PageResult(
            list=[KnowledgeBaseOut.model_validate(item) for item in page.list],
            total=page.total,
            page=page.page,
            pageSize=page.pageSize,
        )

    # 查询单个知识库详情
    async def get(self, knowledge_base_id: UUID) -> KnowledgeBaseOut:
        kb = await self.repo.get_by_id(knowledge_base_id)
        if not kb:
            raise NotFoundError("知识库不存在")
        return KnowledgeBaseOut.model_validate(kb)

    # 创建知识库
    async def create(self, req: KnowledgeBaseCreate, creator_id: UUID | None) -> KnowledgeBaseOut:
        if await self.repo.get_by_code(req.code):
            raise BizError("知识库编码已存在")
        await self._validate_embedding_model(req.embedding_model_id)
        kb = KnowledgeBase(
            name=req.name,
            code=req.code,
            description=req.description,
            icon_id=req.icon_id,
            organization_id=req.organization_id,
            embedding_model_id=req.embedding_model_id,
            status=1,
            creator_id=creator_id,
        )
        return KnowledgeBaseOut.model_validate(await self.repo.create(kb))

    # 更新知识库（code 不可改）
    async def update(self, knowledge_base_id: UUID, req: KnowledgeBaseUpdate) -> KnowledgeBaseOut:
        kb = await self.repo.get_by_id(knowledge_base_id)
        if not kb:
            raise NotFoundError("知识库不存在")
        if req.name is not None:
            kb.name = req.name
        if req.description is not None:
            kb.description = req.description
        if req.icon_id is not None:
            kb.icon_id = req.icon_id
        if req.organization_id is not None:
            kb.organization_id = req.organization_id
        if req.embedding_model_id is not None:
            await self._validate_embedding_model(req.embedding_model_id)
            kb.embedding_model_id = req.embedding_model_id
        if req.status is not None:
            kb.status = req.status
        return KnowledgeBaseOut.model_validate(await self.repo.update(kb))

    # 删除知识库（文档/切分块由 DB CASCADE 删除；同时清理 Qdrant 向量）
    async def delete(self, knowledge_base_id: UUID) -> None:
        kb = await self.repo.get_by_id(knowledge_base_id)
        if not kb:
            raise NotFoundError("知识库不存在")
        # 清理 Qdrant 向量（按 knowledge_base_id payload 过滤）
        from app.modules.ai import vector

        await vector.delete_by_filter(knowledge_base_id=kb.id.hex)
        await self.repo.delete(kb)

    # 上传文档：复用文件上传 → 建文档记录(pending) → 入队 ARQ 后台向量化
    async def upload_document(
        self, file: UploadFile, knowledge_base_id: UUID, uploader_id: UUID
    ) -> DocumentOut:
        kb = await self.repo.get_by_id(knowledge_base_id)
        if not kb:
            raise NotFoundError("知识库不存在")
        if kb.status != 1:
            raise BizError("知识库已禁用，无法上传文档")
        # 复用通用文件上传（biz_type=knowledge_doc）
        file_out = await FileService(self.db).upload(file, uploader_id, FileBizType.KNOWLEDGE_DOC)
        doc = KnowledgeDocument(
            knowledge_base_id=knowledge_base_id,
            file_id=file_out.id,
            name=file_out.name,
            status=DocumentStatus.PENDING.value,
            chunk_count=0,
            uploader_id=uploader_id,
        )
        doc = await self.doc_repo.create(doc)
        # 入队后台向量化（失败不影响上传，任务内会标记 failed）
        from app.tasks.worker import enqueue_job

        try:
            await enqueue_job("process_document", doc.id.hex)
        except Exception as exc:  # noqa: BLE001 - 入队失败仅记录，不阻断上传
            logger.warning("文档入队失败 document_id={} err={}", doc.id, exc)
        return DocumentOut.model_validate(doc)

    # 删除文档：清理 Qdrant 向量 + 删除记录（切分块由 DB CASCADE 删除）
    async def delete_document(self, document_id: UUID) -> None:
        doc = await self.doc_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundError("文档不存在")
        from app.modules.ai import vector

        await vector.delete_by_filter(document_id=doc.id.hex)
        await self.doc_repo.delete(doc)

    # 重试文档向量化：复用原文件重新入队（仅 failed / pending 可重试）
    async def reprocess_document(self, document_id: UUID) -> DocumentOut:
        doc = await self.doc_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundError("文档不存在")
        if doc.status == DocumentStatus.PARSING.value:
            raise BizError("文档正在处理中，请勿重复操作")
        if doc.status == DocumentStatus.PARSED.value:
            raise BizError("文档已处理完成，无需重试")
        # 重置状态并重新入队（复用原文件，不重新上传）
        doc.status = DocumentStatus.PENDING.value
        doc.error_message = None
        doc.chunk_count = 0
        await self.doc_repo.update(doc)
        from app.tasks.worker import enqueue_job

        try:
            await enqueue_job("process_document", doc.id.hex)
        except Exception as exc:  # noqa: BLE001 - 入队失败仅记录，不阻断重试
            logger.warning("文档重试入队失败 document_id={} err={}", doc.id, exc)
        return DocumentOut.model_validate(doc)

    # 分页查询某知识库的文档列表（knowledge_base_id 从查询入参 body 传入）
    async def list_documents(self, query: DocumentQuery) -> PageResult[DocumentOut]:
        kb = await self.repo.get_by_id(query.knowledge_base_id)
        if not kb:
            raise NotFoundError("知识库不存在")
        page_params = PageParams(page=query.page, pageSize=query.pageSize)
        page = await self.doc_repo.list_page(
            query.knowledge_base_id,
            page_params,
            keyword=query.keyword,
            status=query.status.value if query.status else None,
        )
        return PageResult(
            list=[DocumentOut.model_validate(item) for item in page.list],
            total=page.total,
            page=page.page,
            pageSize=page.pageSize,
        )

    # 查询某文档的全部分块（按序号升序）
    async def list_document_chunks(self, document_id: UUID) -> list[ChunkOut]:
        doc = await self.doc_repo.get_by_id(document_id)
        if not doc:
            raise NotFoundError("文档不存在")
        chunks = await self.chunk_repo.list_by_document(document_id)
        return [ChunkOut.model_validate(c) for c in chunks]

    # 按关键字搜索某知识库内的分块（含来源文档名，分页）
    async def search_chunks(self, query: ChunkSearchQuery) -> PageResult[ChunkSearchOut]:
        kb = await self.repo.get_by_id(query.knowledge_base_id)
        if not kb:
            raise NotFoundError("知识库不存在")
        page_params = PageParams(page=query.page, pageSize=query.pageSize)
        page = await self.chunk_repo.search_by_keyword(
            query.knowledge_base_id, query.keyword, page_params
        )
        items: list[ChunkSearchOut] = []
        for chunk, doc_name in page.list:
            out = ChunkSearchOut.model_validate(chunk)
            out.document_name = doc_name
            items.append(out)
        return PageResult(
            list=items,
            total=page.total,
            page=page.page,
            pageSize=page.pageSize,
        )

    # 校验 embedding 模型实例：存在、启用且类型为 embedding
    async def _validate_embedding_model(self, instance_id: UUID | None) -> None:
        if instance_id is None:
            return
        instance = await self.instance_repo.get_by_id(instance_id)
        if instance is None:
            raise NotFoundError("绑定模型实例不存在")
        if instance.model_type != ModelType.EMBEDDING.value:
            raise ValidateError("绑定模型实例必须为 embedding 类型")
        if instance.status != 1:
            raise BizError("绑定模型实例已停用")


async def process_document_task(document_id: str) -> dict:
    """ARQ 后台任务：解析 → 切分 → embedding → 写 Qdrant → 双写 MySQL chunk。

    独立 AsyncSessionLocal 会话（worker 进程内执行，不依赖请求上下文）。
    状态流转：pending → parsing → parsed / failed。
    """
    from app.core.database import AsyncSessionLocal
    from app.modules.ai import vector
    from app.modules.knowledge.parser import parse_document, split_text

    async with AsyncSessionLocal() as db:
        doc_repo = DocumentRepository(db)
        kb_repo = KnowledgeBaseRepository(db)
        chunk_repo = ChunkRepository(db)
        file_service = FileService(db)

        doc = await doc_repo.get_by_id(UUID(document_id))
        if not doc:
            return {"ok": False, "error": "文档不存在"}

        doc.status = DocumentStatus.PARSING.value
        await doc_repo.update(doc)
        try:
            kb = await kb_repo.get_by_id(doc.knowledge_base_id)
            if not kb:
                raise BizError("知识库不存在")

            # 读取文件内容（本地路径或字节）
            content = await file_service.get_content(doc.file_id)
            if content.data is not None:
                data = content.data
            elif content.path is not None:
                data = content.path.read_bytes()
            else:
                raise BizError("文件内容不可读")

            # 解析 + 切分
            ext = doc.name.rsplit(".", 1)[-1].lower() if "." in doc.name else None
            text = parse_document(data, ext)
            chunks = split_text(text)
            if not chunks:
                raise BizError("文档内容为空或无法切分")

            # embedding（读知识库绑定的 embedding 模型实例）
            embeddings = await vector.build_embeddings(db, kb.embedding_model_id)
            vectors = await embeddings.aembed_documents(chunks)
            dim = len(vectors[0])
            await vector.ensure_collection(dim)

            # 写 Qdrant + 双写 MySQL chunk
            points: list[dict] = []
            chunk_records: list[KnowledgeChunk] = []
            for i, (chunk_text, vec) in enumerate(zip(chunks, vectors, strict=False), start=1):
                chunk_id = uuid4()
                point_id = str(chunk_id)
                points.append(
                    {
                        "id": point_id,
                        "vector": vec,
                        "payload": {
                            "knowledge_base_id": doc.knowledge_base_id.hex,
                            "document_id": doc.id.hex,
                            "chunk_id": chunk_id.hex,
                            "seq_no": i,
                            "content": chunk_text,
                        },
                    }
                )
                chunk_records.append(
                    KnowledgeChunk(
                        document_id=doc.id,
                        knowledge_base_id=doc.knowledge_base_id,
                        seq_no=i,
                        content=chunk_text,
                        token_count=max(1, len(chunk_text) // 2),  # 粗略估算（中文为主）
                        vector_id=point_id,
                        char_count=len(chunk_text),
                    )
                )
            await vector.upsert_chunks(points)
            await chunk_repo.create_many(chunk_records)

            doc.status = DocumentStatus.PARSED.value
            doc.chunk_count = len(chunk_records)
            await doc_repo.update(doc)
            logger.info("文档处理完成 document_id={} chunks={}", doc.id, len(chunk_records))
            return {"ok": True, "chunk_count": len(chunk_records)}
        except Exception as exc:  # noqa: BLE001 - 任务内兜底，标记 failed
            logger.exception("文档处理失败 document_id={}", document_id)
            doc.status = DocumentStatus.FAILED.value
            doc.error_message = str(exc)[:500]
            await doc_repo.update(doc)
            return {"ok": False, "error": str(exc)}

