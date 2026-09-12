"""AI 能力数据模型（SQLAlchemy ORM）。

会话持久化：AiConversation（对话会话）+ AiMessage（会话内消息）。
对话记录落库，支持多轮记忆与历史查询。
"""

from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base, TimestampMixin


class AiConversation(TimestampMixin, Base):
    """AI 对话会话。"""

    __tablename__ = "ai_conversation"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # 所属 Agent（删除 Agent 级联删除会话）
    agent_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("sys_agent.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # 发起用户（用户删除后保留，置空）
    user_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("sys_user.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # 会话标题（默认取首条用户消息截断）
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="新会话")

    # 状态（active / archived）
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")


class AiMessage(TimestampMixin, Base):
    """会话内消息。"""

    __tablename__ = "ai_message"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # 所属会话（删除会话级联删除消息）
    conversation_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("ai_conversation.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # 角色（user / assistant）
    role: Mapped[str] = mapped_column(String(16), nullable=False)

    # 消息内容
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # 模型推理内容（思维链，推理型模型才有；可空）
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
