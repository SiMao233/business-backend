"""模型管理数据模型（SQLAlchemy ORM）。

模型管理采用「供应商 → 实例」两级结构：
- ModelProvider（sys_model_provider）：模型供应商（OpenAI / DeepSeek / 通义等）；
- ModelInstance（sys_model_instance）：供应商下的具体模型实例，Agent 通过 model_id 绑定。
"""

from uuid import UUID, uuid4

from sqlalchemy import Boolean, ForeignKey, Integer, SmallInteger, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base, TimestampMixin


class ModelProvider(TimestampMixin, Base):
    """模型供应商。"""

    __tablename__ = "sys_model_provider"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # 供应商名称（展示用）
    name: Mapped[str] = mapped_column(String(64), nullable=False)

    # 业务编码（全局唯一，如 openai / deepseek）
    code: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)

    # OpenAI 兼容协议的自定义地址（为空则用官方默认）
    base_url: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # API Key（TODO: 生产环境需加密存储，出参不回传，仅返回 has_api_key）
    api_key: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # 是否内置供应商（由 seed 初始化，禁止删除）
    is_builtin: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, server_default="0"
    )

    # 状态：1 启用 / 0 禁用
    status: Mapped[int] = mapped_column(SmallInteger, default=1, nullable=False)

    # 该供应商下的模型实例
    instances = relationship("ModelInstance", back_populates="provider")


class ModelInstance(TimestampMixin, Base):
    """模型实例（Agent 实际绑定对象）。"""

    __tablename__ = "sys_model_instance"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # 所属供应商
    provider_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("sys_model_provider.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # 展示名（如 GPT-4o / deepseek-chat）
    name: Mapped[str] = mapped_column(String(64), nullable=False)

    # 调用时使用的模型标识（如 gpt-4o / deepseek-chat）
    code: Mapped[str] = mapped_column(String(64), nullable=False)

    # 模型类型（ModelType 枚举值：chat / embedding / image / audio）
    model_type: Mapped[str] = mapped_column(String(16), nullable=False, default="chat")

    # 上下文上限（token）
    max_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # 状态：1 启用 / 0 禁用
    status: Mapped[int] = mapped_column(SmallInteger, default=1, nullable=False)

    provider = relationship("ModelProvider", back_populates="instances")
