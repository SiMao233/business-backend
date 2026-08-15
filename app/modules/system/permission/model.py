"""系统权限管理数据模型（SQLAlchemy ORM）。

权限采用「树状组织 + 扁平权限码」双重模型：
- 树状组织：`parent_id` 自引用构成目录/菜单/按钮层级，用于管理端树形展示与分配；
- 扁平权限码：`code` 全局唯一（如 system:user:create），鉴权时直接比对 code。
"""

from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, SmallInteger, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base, TimestampMixin
from app.modules.iam.model import RolePermission  # secondary 使用其底层 Table：RolePermission.__table__


# 权限表
class Permission(
    TimestampMixin,
    Base
):

    __tablename__ = "sys_permission"

    id: Mapped[UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid4
    )

    # 上级权限 ID（自引用，根节点为 NULL；树状组织用）
    parent_id: Mapped[UUID | None] = mapped_column(
        Uuid,
        ForeignKey("sys_permission.id", ondelete="CASCADE"),
        nullable=True,
        index=True
    )

    # 权限码（全局唯一，扁平，如 system:user:create）
    code: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    # 类型：1=目录 2=菜单 3=按钮
    type: Mapped[int] = mapped_column(
        SmallInteger,
        default=1
    )

    description: Mapped[str | None] = mapped_column(
        String(255)
    )

    status: Mapped[int] = mapped_column(
        SmallInteger,
        default=1
    )

    roles = relationship(
        "Role",
        secondary=RolePermission.__table__,
        back_populates="permissions",
        lazy="selectin"
    )

    # 子权限。注意：自引用关系在此 SQLAlchemy 版本下 lazy="selectin" 不生效（会退化为
    # 逐条惰性加载，异步下报 MissingGreenlet），需要子权限时请显式
    # `selectinload(Permission.children)`（见 PermissionRepository.list_all）。
    children = relationship(
        "Permission",
        back_populates="parent"
    )

    parent = relationship(
        "Permission",
        remote_side="Permission.id",
        back_populates="children"
    )
