"""Agent 数据模型（SQLAlchemy ORM）。

Agent 配置管理：Agent（sys_agent）定义 + AgentVersion（sys_agent_version）版本快照。
运行逻辑由独立 ai-service 承担，本模块仅管理配置与发布。
"""

from uuid import UUID, uuid4

from sqlalchemy import JSON, ForeignKey, Integer, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin


class Agent(TimestampMixin, Base):
    """Agent 定义。"""

    __tablename__ = "sys_agent"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # 名称
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # 业务编码（全局唯一，ai-service 识别用）
    code: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)

    description: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # 图标（sys_file，biz_type=agent_attachment）
    icon_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("sys_file.id", ondelete="SET NULL"), nullable=True
    )

    # 绑定模型实例（sys_model_instance）
    model_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("sys_model_instance.id", ondelete="SET NULL"), nullable=True
    )

    # 归属组织（sys_organization）
    organization_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("sys_organization.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # 核心配置（JSON：system_prompt / temperature / tools 等，结构见 AgentConfig）
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # 生命周期状态（AgentStatus：draft / running / paused / stopped）
    status: Mapped[str] = mapped_column(String(16), default="draft", nullable=False)

    # 当前发布版本号（0=未发布）
    current_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # 创建人（用户删除后保留，置空）
    creator_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("sys_user.id", ondelete="SET NULL"), nullable=True
    )


class AgentVersion(TimestampMixin, Base):
    """Agent 配置版本快照（发布后不可变）。"""

    __tablename__ = "sys_agent_version"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    agent_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("sys_agent.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # 自增版本号（1,2,3...）
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    # 该版本配置快照
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    changelog: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # 版本状态（AgentVersionStatus：draft / published）
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")
