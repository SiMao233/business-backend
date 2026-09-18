"""知识库数据模型（SQLAlchemy ORM）。

三级结构支撑 RAG：
- KnowledgeBase（sys_knowledge_base）：知识库容器，绑定组织与 embedding 模型；
- KnowledgeDocument（sys_knowledge_document）：知识库下的文档（对应 sys_file 上传文件）；
- KnowledgeChunk（sys_knowledge_chunk）：文档切分后的文本块，向量落 Qdrant，
  vector_id 关联 Qdrant point ID（MySQL 存文本元数据，便于浏览/审计/回源）。
"""

from uuid import UUID, uuid4

from sqlalchemy import BigInteger, ForeignKey, Integer, SmallInteger, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin


class KnowledgeBase(TimestampMixin, Base):
    """知识库容器。"""

    __tablename__ = "sys_knowledge_base"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # 名称
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # 业务编码（全局唯一，Agent 绑定引用）
    code: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)

    description: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # 图标（sys_file）
    icon_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("sys_file.id", ondelete="SET NULL"), nullable=True
    )

    # 归属组织（sys_organization）
    organization_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("sys_organization.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # 绑定 embedding 模型实例（sys_model_instance，service 层校验 model_type=embedding）
    embedding_model_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("sys_model_instance.id", ondelete="SET NULL"), nullable=True
    )

    # 状态：1 启用 / 0 禁用
    status: Mapped[int] = mapped_column(SmallInteger, default=1, nullable=False)

    # 创建人（用户删除后保留，置空）
    creator_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("sys_user.id", ondelete="SET NULL"), nullable=True
    )


class KnowledgeDocument(TimestampMixin, Base):
    """知识库文档（对应 sys_file 上传文件，切分后产生 chunk）。"""

    __tablename__ = "sys_knowledge_document"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # 所属知识库（删除知识库级联删除文档）
    knowledge_base_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("sys_knowledge_base.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # 源文件（sys_file；文件被删后保留记录，置空）
    file_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("sys_file.id", ondelete="SET NULL"), nullable=True
    )

    # 文档名（原始文件名）
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # 处理状态（DocumentStatus：pending / parsing / parsed / failed）
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")

    # 切分出的 chunk 数量（解析完成后回填）
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # 解析失败原因
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # 上传者（用户删除后保留，置空）
    uploader_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("sys_user.id", ondelete="SET NULL"), nullable=True
    )


class KnowledgeChunk(TimestampMixin, Base):
    """文档切分后的文本块。

    文本内容与元数据落 MySQL（双写方案）；向量数据存 Qdrant，
    vector_id 关联 Qdrant point ID，便于删除/回源。
    """

    __tablename__ = "sys_knowledge_chunk"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # 所属文档（删除文档级联删除 chunk）
    document_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("sys_knowledge_document.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # 所属知识库（冗余存，Qdrant payload 过滤 / 按知识库清理用）
    knowledge_base_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("sys_knowledge_base.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # 文档内序号（从 1 开始）
    seq_no: Mapped[int] = mapped_column(Integer, nullable=False)

    # 切分文本内容
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # 章节路径（结构感知切分产出，如「员工手册 > 二、考勤」；无标题结构时为 NULL）
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # 来源页码（PDF 等分页文档；跨页块取起始页）
    page_no: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # 块类型（ChunkType：text / table / mixed）
    chunk_type: Mapped[str] = mapped_column(String(16), nullable=False, default="text")

    # 预估 token 数（可选，向量化后回填）
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Qdrant point ID（向量写入后回填，双写关联）
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 字符数（展示/统计用）
    char_count: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
