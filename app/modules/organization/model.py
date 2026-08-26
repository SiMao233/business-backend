"""组织管理数据模型（SQLAlchemy ORM）。

组织采用树状结构（parent_id 自引用），支持多级组织归属。
Agent 通过 organization_id 归属到组织。
"""

from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, SmallInteger, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base, TimestampMixin


class Organization(TimestampMixin, Base):
    """组织。"""

    __tablename__ = "sys_organization"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # 上级组织 ID（自引用，根节点为 NULL；树状组织用）
    parent_id: Mapped[UUID | None] = mapped_column(
        Uuid,
        ForeignKey("sys_organization.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # 组织名称
    name: Mapped[str] = mapped_column(String(64), nullable=False)

    # 业务编码（全局唯一）
    code: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)

    description: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # 负责人（用户删除后保留组织，置空）
    owner_id: Mapped[UUID | None] = mapped_column(
        Uuid,
        ForeignKey("sys_user.id", ondelete="SET NULL"),
        nullable=True,
    )

    # 状态：1 启用 / 0 禁用
    status: Mapped[int] = mapped_column(SmallInteger, default=1, nullable=False)

    # 子组织。注意：自引用关系 lazy="selectin" 不生效，需显式 selectinload（见 OrganizationRepository.list_all）
    children = relationship("Organization", back_populates="parent")

    parent = relationship("Organization", remote_side="Organization.id", back_populates="children")
