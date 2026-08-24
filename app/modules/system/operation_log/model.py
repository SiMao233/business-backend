"""系统操作日志数据模型（SQLAlchemy ORM）。

记录后台管理的高风险操作（删除、分配权限等）用于审计追溯：
谁在何时对什么对象做了什么操作、结果如何。
"""

from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, SmallInteger, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin


class OperationLog(TimestampMixin, Base):
    """操作日志表：追加写入，仅记录不修改。"""

    __tablename__ = "sys_operation_log"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # 操作人（外键 SET NULL：用户删除后日志仍保留，靠 username 快照追溯）
    user_id: Mapped[UUID | None] = mapped_column(
        Uuid,
        ForeignKey("sys_user.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    username: Mapped[str | None] = mapped_column(String(50))

    # 操作对象：模块 + 动作 + 目标 ID
    module: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    target_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)

    # 请求上下文
    method: Mapped[str | None] = mapped_column(String(10))
    path: Mapped[str | None] = mapped_column(String(255))
    ip: Mapped[str | None] = mapped_column(String(50))

    # 变更摘要（JSON 字符串）与结果
    detail: Mapped[str | None] = mapped_column(Text)
    status: Mapped[int] = mapped_column(SmallInteger, default=1, nullable=False)  # 1=成功 0=失败
    error_msg: Mapped[str | None] = mapped_column(String(500))