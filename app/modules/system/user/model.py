"""系统用户管理数据模型（SQLAlchemy ORM）。

TODO(user): 定义用户管理相关表模型（若复用 iam 的 users 表则无需新建），
并纳入 Alembic 自动迁移。
"""

# 用户-角色 多对多关联表
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Boolean, SmallInteger, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base, TimestampMixin
from app.modules.iam.model import UserRole  # secondary 使用其底层 Table：UserRole.__table__


# 用户表
class User(TimestampMixin, Base):
    __tablename__ = "sys_user"

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4
    )


    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        index=True,
        nullable=False
    )


    email: Mapped[Optional[str]] = mapped_column(
        String(100),
        unique=True,
        nullable=True
    )

    phone: Mapped[Optional[str]] = mapped_column(
        String(20)
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    nickname: Mapped[Optional[str]] = mapped_column(
        String(50)
    )


    avatar: Mapped[Optional[str]] = mapped_column(
        String(500)
    )

    status: Mapped[int] = mapped_column(
        SmallInteger,
        default=1
    )

    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False
    )

    # 登录信息
    last_login_ip: Mapped[Optional[str]] = mapped_column(
        String(50)
    )

    # 关系
    roles = relationship(
        "Role",
        secondary=UserRole.__table__,
        back_populates="users"
    )