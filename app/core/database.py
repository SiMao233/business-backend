"""数据库基础设施：异步引擎、会话工厂与依赖注入。"""

from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from sqlalchemy import DateTime, func
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.config import Settings, get_settings

def _now() -> datetime:
    """时间列默认值：应用进程生成（避免 SQL 默认值导致提交后列过期）。

    时间基准统一为 UTC，MySQL DATETIME 列存 UTC 墙钟时间；
    接口输出同样为 UTC，前端负责本地化展示。
    """
    return datetime.now(timezone.utc)

class Base(DeclarativeBase):
    """所有 ORM 模型的声明式基类。"""
    pass

class TimestampMixin:
    """公共时间字段"""
    # 创建时间
    create_time: Mapped[datetime] = mapped_column(
        DateTime, 
        default=_now, 
        nullable=False
    )
    # 更新时间
    update_time: Mapped[datetime] = mapped_column(
        DateTime, 
        default=_now, 
        onupdate=_now, 
        nullable=False
    )

def create_engine(settings: Settings | None = None) -> AsyncEngine:
    """根据配置创建 MySQL 异步引擎。"""
    settings = settings or get_settings()
    return create_async_engine(
        settings.database_url,
        echo=settings.db_echo,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=True,
    )


# 全局引擎与会话工厂（连接池由应用 lifespan 负责关闭）
engine: AsyncEngine = create_engine()

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def dispose_engine() -> None:
    """释放全局引擎连接池（应用关闭时调用）。"""
    await engine.dispose()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：提供请求级数据库会话，请求结束自动关闭。

    API -> Service -> Repository 分层中，会话由 API 层通过依赖注入获得，
    并在请求结束时统一提交/回滚与关闭。
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
