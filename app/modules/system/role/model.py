"""系统角色管理数据模型（SQLAlchemy ORM）。

TODO(role): 定义角色管理相关表模型（若复用 iam 的 roles 表则无需新建），
并纳入 Alembic 自动迁移。
"""

from uuid import UUID, uuid4

from sqlalchemy import SmallInteger, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base, TimestampMixin
from app.modules.iam.model import RolePermission, UserRole  # secondary 使用其底层 Table


# 角色表
class Role(
    TimestampMixin,
    Base
):

    __tablename__ = "sys_role"

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4
    )

    code: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False
    )

    name: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )

    description: Mapped[str | None] = mapped_column(
        String(255)
    )

    status: Mapped[int] = mapped_column(
        SmallInteger,
        default=1
    )

    users = relationship(
        "User",
        secondary=UserRole.__table__,
        back_populates="roles"
    )

    permissions = relationship(
        "Permission",
        secondary=RolePermission.__table__,
        back_populates="roles",
        lazy="selectin"
    )
