"""用量统计数据模型（SQLAlchemy ORM）。

`AiUsageRecord`（ai_usage_record）：模型调用用量明细，每次模型调用一条（追加写入，不修改）。
- 工具模式一次回答可能有多次 LLM 调用 → 多条例；
- 维度字段冗余「名称快照」，源数据（用户 / Agent / 模型实例）删除后仍可追溯，统计查询免 JOIN；
- `usage_date` 在写入时按业务时区（settings.usage_timezone）预计算，供按天分组，
  避免运行时 CONVERT_TZ 依赖 MySQL 时区表。
"""

from datetime import date
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import Date, Index, Integer, Numeric, SmallInteger, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin


class AiUsageRecord(TimestampMixin, Base):
    """模型调用用量明细（追加写入，不修改）。"""

    __tablename__ = "ai_usage_record"
    __table_args__ = (
        Index("ix_ai_usage_record_usage_date_user_id", "usage_date", "user_id"),
        Index("ix_ai_usage_record_usage_date_agent_id", "usage_date", "agent_id"),
        Index("ix_ai_usage_record_usage_date_organization_id", "usage_date", "organization_id"),
        Index(
            "ix_ai_usage_record_usage_date_model_instance_id", "usage_date", "model_instance_id"
        ),
        Index("ix_ai_usage_record_create_time", "create_time"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # ---- 统计维度（不建 FK：源数据删除不阻塞写入，靠快照追溯）----
    user_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True, index=True)
    username: Mapped[str | None] = mapped_column(String(50), nullable=True)
    agent_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True, index=True)
    agent_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    organization_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True, index=True)
    model_instance_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True, index=True)
    provider_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True, index=True)
    provider_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    conversation_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)
    message_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)

    # ---- 用量 ----
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # ---- 成本（单价快照，单位：元 / 千 token）----
    input_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    output_price: Mapped[Decimal | None] = mapped_column(Numeric(18, 6), nullable=True)
    total_cost: Mapped[Decimal] = mapped_column(Numeric(18, 6), default=0, nullable=False)

    # ---- 上下文 ----
    # 调用类型：chat（对话）/ tool（工具模式）/ embedding（向量化，第二期接入）
    call_type: Mapped[str] = mapped_column(String(16), default="chat", nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[int] = mapped_column(SmallInteger, default=1, nullable=False)  # 1成功 0失败
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 业务自然日（按 usage_timezone 切分，供按天分组）
    usage_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
