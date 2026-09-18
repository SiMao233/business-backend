"""知识库 Pydantic Schema：请求 / 响应模型。"""

from uuid import UUID

from pydantic import ConfigDict, Field, field_serializer

from app.common.enums import DocumentStatus
from app.common.schema import ApiInModel, ApiOutModel, UtcDateTime


class KnowledgeBaseCreate(ApiInModel):
    """创建知识库入参。"""

    name: str = Field(min_length=1, max_length=255, description="知识库名称")
    code: str = Field(min_length=1, max_length=64, description="业务编码（全局唯一）")
    description: str | None = Field(default=None, max_length=512, description="描述")
    icon_id: UUID | None = Field(default=None, description="图标文件 ID")
    organization_id: UUID | None = Field(default=None, description="归属组织 ID")
    embedding_model_id: UUID | None = Field(
        default=None, description="绑定 embedding 模型实例 ID（model_type=embedding）"
    )


class KnowledgeBaseUpdate(ApiInModel):
    """更新知识库入参（code 不可改）。"""

    name: str | None = Field(default=None, min_length=1, max_length=255, description="知识库名称")
    description: str | None = Field(default=None, max_length=512, description="描述")
    icon_id: UUID | None = Field(default=None, description="图标文件 ID")
    organization_id: UUID | None = Field(default=None, description="归属组织 ID")
    embedding_model_id: UUID | None = Field(
        default=None, description="绑定 embedding 模型实例 ID（model_type=embedding）"
    )
    status: int | None = Field(default=None, ge=0, le=1, description="状态（1 启用 / 0 禁用）")


class KnowledgeBaseOut(ApiOutModel):
    """知识库出参。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    code: str
    description: str | None = None
    icon_id: UUID | None = None
    organization_id: UUID | None = None
    embedding_model_id: UUID | None = None
    status: int
    creator_id: UUID | None = None
    create_time: UtcDateTime
    update_time: UtcDateTime

    @field_serializer("id", "icon_id", "organization_id", "embedding_model_id", "creator_id")
    def serialize_id(self, value: UUID | None) -> str | None:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex if value else None


class KnowledgeBaseQuery(ApiInModel):
    """知识库分页查询。"""

    page: int = Field(default=1, ge=1, description="页码（从 1 开始）")
    pageSize: int = Field(default=20, ge=10, le=100, description="每页条数")
    keyword: str | None = Field(default=None, max_length=64, description="关键字：名称/编码模糊匹配")
    organization_id: UUID | None = Field(default=None, description="按组织筛选")
    status: int | None = Field(default=None, ge=0, le=1, description="按状态筛选")


class DocumentOut(ApiOutModel):
    """知识库文档出参。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    knowledge_base_id: UUID
    file_id: UUID | None = None
    name: str
    status: DocumentStatus
    chunk_count: int
    error_message: str | None = None
    uploader_id: UUID | None = None
    create_time: UtcDateTime
    update_time: UtcDateTime

    @field_serializer("id", "knowledge_base_id", "file_id", "uploader_id")
    def serialize_id(self, value: UUID | None) -> str | None:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex if value else None


class DocumentQuery(ApiInModel):
    """文档分页查询。"""

    knowledge_base_id: UUID = Field(description="知识库 ID")
    page: int = Field(default=1, ge=1, description="页码（从 1 开始）")
    pageSize: int = Field(default=20, ge=10, le=100, description="每页条数")
    keyword: str | None = Field(default=None, max_length=64, description="关键字：文档名模糊匹配")
    status: DocumentStatus | None = Field(default=None, description="按处理状态筛选")


class ChunkOut(ApiOutModel):
    """文档分块出参。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    knowledge_base_id: UUID
    seq_no: int
    content: str
    section_path: str | None = Field(
        default=None, description="所属章节路径（无标题结构时为 null）"
    )
    page_no: int | None = Field(default=None, description="来源页码（分页文档，如 PDF）")
    chunk_type: str = Field(default="text", description="块类型：text / table / mixed")
    token_count: int | None = None
    char_count: int | None = None
    vector_id: str | None = None
    create_time: UtcDateTime

    @field_serializer("id", "document_id", "knowledge_base_id")
    def serialize_id(self, value: UUID) -> str:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex


class ChunkSearchOut(ApiOutModel):
    """分块搜索出参（含来源文档名）。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    document_name: str = Field(default="", description="来源文档名")
    knowledge_base_id: UUID
    seq_no: int
    content: str
    section_path: str | None = Field(
        default=None, description="所属章节路径（无标题结构时为 null）"
    )
    page_no: int | None = Field(default=None, description="来源页码（分页文档，如 PDF）")
    chunk_type: str = Field(default="text", description="块类型：text / table / mixed")
    token_count: int | None = None
    char_count: int | None = None
    create_time: UtcDateTime

    @field_serializer("id", "document_id", "knowledge_base_id")
    def serialize_id(self, value: UUID) -> str:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex


class ChunkSearchQuery(ApiInModel):
    """分块关键字搜索入参。"""

    knowledge_base_id: UUID = Field(description="知识库 ID")
    keyword: str = Field(min_length=1, max_length=100, description="搜索关键字")
    page: int = Field(default=1, ge=1, description="页码（从 1 开始）")
    pageSize: int = Field(default=20, ge=10, le=100, description="每页条数")

