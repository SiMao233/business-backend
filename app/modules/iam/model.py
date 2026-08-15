"""IAM 数据模型（SQLAlchemy ORM）。

TODO(auth): 定义 User、Session 等表模型，并纳入 Alembic 自动迁移。
"""

from uuid import UUID

from sqlalchemy import ForeignKey, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin

# 用户-角色关联表
class UserRole(
    TimestampMixin,
    Base
):
    __tablename__ = "sys_user_role"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    user_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey(
            "sys_user.id",
            ondelete="CASCADE"
        )
    )

    role_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey(
            "sys_role.id",
            ondelete="CASCADE"
        )
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "role_id",
            name="uk_user_role"
        ),
    )

# 角色权限关联表
class RolePermission(
    TimestampMixin,
    Base
):

    __tablename__ = "sys_role_permission"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    role_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey(
            "sys_role.id",
            ondelete="CASCADE"
        )
    )

    permission_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey(
            "sys_permission.id",
            ondelete="CASCADE"
        )
    )

    __table_args__ = (
        UniqueConstraint(
            "role_id",
            "permission_id",
            name="uk_role_permission"
        ),
    )