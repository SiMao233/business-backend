"""文件数据模型（SQLAlchemy ORM）。

通用文件模块：所有业务（知识库文档、头像、附件等）共用的文件元数据表 `sys_file`。
实际文件内容存储在存储后端（本地磁盘 / 对象存储），表中仅记录元数据与存储 key。
"""

from uuid import UUID, uuid4

from sqlalchemy import BigInteger, ForeignKey, SmallInteger, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin


class SysFile(TimestampMixin, Base):
    """文件元数据表。"""

    __tablename__ = "sys_file"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # 原始文件名（仅展示用，存储路径用 storage_key）
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # 存储 key（YYYY/MM/{uuid}.{ext}），服务端生成，唯一
    storage_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    # MIME 类型
    content_type: Mapped[str] = mapped_column(String(127), nullable=False)

    # 文件大小（字节）
    size: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # 扩展名（不含点，可为空）
    ext: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # 内容 SHA-256（去重 / 完整性校验预留）
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 上传者（用户删除后保留文件，置空）
    uploader_id: Mapped[UUID | None] = mapped_column(
        Uuid,
        ForeignKey("sys_user.id", ondelete="SET NULL"),
        nullable=True,
    )

    # 业务类型（FileBizType 枚举值，如 knowledge_doc / avatar）
    biz_type: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # 状态：1 有效 / 0 逻辑删除
    status: Mapped[int] = mapped_column(SmallInteger, default=1, nullable=False)
