"""Knowledge API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。

本层禁止直接操作数据库或编写复杂业务逻辑。
路由风格：静态路径（/list /create /documents/upload）注册在 /{knowledge_base_id} 之前，
避免被 UUID 参数吞掉。
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageResult
from app.common.response import ApiResponse, success
from app.core.database import get_db
from app.middleware.authentication import CurrentUserDep, get_current_user
from app.middleware.permission import require_permissions
from app.modules.knowledge.codes import PermissionCode
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
from app.modules.knowledge.service import KnowledgeService

router = APIRouter(prefix="/knowledge", tags=["知识库"], dependencies=[Depends(get_current_user)])

DbDep = Annotated[AsyncSession, Depends(get_db)]


# 依赖注入工厂：创建 KnowledgeService 实例（绑定数据库会话）
def get_service(db: DbDep) -> KnowledgeService:
    return KnowledgeService(db)


# ---- 列表 / 创建（静态路径，注册在 /{knowledge_base_id} 之前）----
@router.post(
    "/list",
    response_model=ApiResponse[PageResult[KnowledgeBaseOut]],
    summary="知识库列表",
    dependencies=[Depends(require_permissions(PermissionCode.KNOWLEDGE_LIST))],
)
# 知识库列表：分页查询，支持关键字/组织/状态条件过滤
async def list_knowledge_bases(
    query: KnowledgeBaseQuery,
    service: Annotated[KnowledgeService, Depends(get_service)],
) -> ApiResponse[PageResult[KnowledgeBaseOut]]:
    return success(data=await service.list_page(query))


@router.post(
    "/create",
    response_model=ApiResponse[KnowledgeBaseOut],
    summary="创建知识库",
    dependencies=[Depends(require_permissions(PermissionCode.KNOWLEDGE_CREATE))],
)
# 创建知识库：校验参数并入库，返回新创建的知识库信息
async def create_knowledge_base(
    req: KnowledgeBaseCreate,
    current: CurrentUserDep,
    service: Annotated[KnowledgeService, Depends(get_service)],
) -> ApiResponse[KnowledgeBaseOut]:
    return success(data=await service.create(req, current.user_id), message="创建成功")


# ---- 详情 / 更新 / 删除（id 放最后）----
@router.get(
    "/{knowledge_base_id}",
    response_model=ApiResponse[KnowledgeBaseOut],
    summary="知识库详情",
    dependencies=[Depends(require_permissions(PermissionCode.KNOWLEDGE_LIST))],
)
# 知识库详情：按 ID 查询单个知识库
async def get_knowledge_base(
    knowledge_base_id: UUID,
    service: Annotated[KnowledgeService, Depends(get_service)],
) -> ApiResponse[KnowledgeBaseOut]:
    return success(data=await service.get(knowledge_base_id))


@router.put(
    "/{knowledge_base_id}",
    response_model=ApiResponse[KnowledgeBaseOut],
    summary="更新知识库",
    dependencies=[Depends(require_permissions(PermissionCode.KNOWLEDGE_UPDATE))],
)
# 更新知识库：按 ID 修改知识库信息
async def update_knowledge_base(
    knowledge_base_id: UUID,
    req: KnowledgeBaseUpdate,
    service: Annotated[KnowledgeService, Depends(get_service)],
) -> ApiResponse[KnowledgeBaseOut]:
    return success(data=await service.update(knowledge_base_id, req), message="更新成功")


@router.delete(
    "/{knowledge_base_id}",
    response_model=ApiResponse[None],
    summary="删除知识库",
    dependencies=[Depends(require_permissions(PermissionCode.KNOWLEDGE_DELETE))],
)
# 删除知识库：按 ID 删除（文档/切分块级联删除）
async def delete_knowledge_base(
    knowledge_base_id: UUID,
    service: Annotated[KnowledgeService, Depends(get_service)],
) -> ApiResponse[None]:
    await service.delete(knowledge_base_id)
    return success(message="删除成功")


# ---- 文档列表（知识库下的资源，列表用 POST + body 查询，遵循项目约定）----
@router.post(
    "/documents/list",
    response_model=ApiResponse[PageResult[DocumentOut]],
    summary="知识库文档列表",
    dependencies=[Depends(require_permissions(PermissionCode.KNOWLEDGE_DOCUMENT_LIST))],
)
# 文档列表：分页查询某知识库下的文档（knowledge_base_id 从 body 传入）
async def list_documents(
    query: DocumentQuery,
    service: Annotated[KnowledgeService, Depends(get_service)],
) -> ApiResponse[PageResult[DocumentOut]]:
    return success(data=await service.list_documents(query))


# ---- 文档上传 / 删除（静态路径 /documents/upload 注册在 /{knowledge_base_id} 之前）----
@router.post(
    "/documents/upload",
    response_model=ApiResponse[DocumentOut],
    summary="上传知识库文档",
    dependencies=[Depends(require_permissions(PermissionCode.KNOWLEDGE_DOCUMENT_UPLOAD))],
)
# 上传文档：multipart 表单，file 必填，knowledgeBaseId 必填；上传后入队后台向量化
async def upload_document(
    file: Annotated[UploadFile, File(description="文档文件")],
    knowledge_base_id: Annotated[UUID, Form(description="知识库 ID")],
    current_user: CurrentUserDep = None,
    service: Annotated[KnowledgeService, Depends(get_service)] = None,
) -> ApiResponse[DocumentOut]:
    return success(
        data=await service.upload_document(file, knowledge_base_id, current_user.user_id),
        message="上传成功，后台向量化处理中",
    )


@router.delete(
    "/documents/{document_id}",
    response_model=ApiResponse[None],
    summary="删除知识库文档",
    dependencies=[Depends(require_permissions(PermissionCode.KNOWLEDGE_DOCUMENT_DELETE))],
)
# 删除文档：清理 Qdrant 向量 + 删除记录（切分块级联删除）
async def delete_document(
    document_id: UUID,
    service: Annotated[KnowledgeService, Depends(get_service)],
) -> ApiResponse[None]:
    await service.delete_document(document_id)
    return success(message="删除成功")


@router.post(
    "/documents/reprocess/{document_id}",
    response_model=ApiResponse[DocumentOut],
    summary="重试文档向量化",
    dependencies=[Depends(require_permissions(PermissionCode.KNOWLEDGE_DOCUMENT_UPLOAD))],
)
# 重试文档：复用原文件重新入队后台向量化（仅 failed / pending 可重试）
async def reprocess_document(
    document_id: UUID,
    service: Annotated[KnowledgeService, Depends(get_service)],
) -> ApiResponse[DocumentOut]:
    return success(
        data=await service.reprocess_document(document_id),
        message="已重新入队，后台向量化处理中",
    )


# ---- 分块查询 / 分块搜索（动作前置、id 放最后）----
@router.get(
    "/documents/chunks/{document_id}",
    response_model=ApiResponse[list[ChunkOut]],
    summary="文档分块列表",
    dependencies=[Depends(require_permissions(PermissionCode.KNOWLEDGE_DOCUMENT_LIST))],
)
# 文档分块列表：按文档 ID 查询该文档的全部分块（按序号升序）
async def list_document_chunks(
    document_id: UUID,
    service: Annotated[KnowledgeService, Depends(get_service)],
) -> ApiResponse[list[ChunkOut]]:
    return success(data=await service.list_document_chunks(document_id))


@router.post(
    "/chunks/search",
    response_model=ApiResponse[PageResult[ChunkSearchOut]],
    summary="知识库分块搜索",
    dependencies=[Depends(require_permissions(PermissionCode.KNOWLEDGE_DOCUMENT_LIST))],
)
# 分块搜索：按关键字搜索某知识库内的全部分块（含来源文档名，分页）
async def search_chunks(
    query: ChunkSearchQuery,
    service: Annotated[KnowledgeService, Depends(get_service)],
) -> ApiResponse[PageResult[ChunkSearchOut]]:
    return success(data=await service.search_chunks(query))
