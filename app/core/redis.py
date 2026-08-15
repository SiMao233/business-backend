"""Redis 基础设施：异步客户端与依赖注入。"""

from collections.abc import AsyncGenerator

from redis.asyncio import Redis

from app.core.config import Settings, get_settings


def create_redis_client(settings: Settings | None = None) -> Redis:
    """根据配置创建 Redis 异步客户端。"""
    settings = settings or get_settings()
    return Redis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)


# 全局 Redis 客户端（连接池由应用 lifespan 负责关闭）
redis_client: Redis = create_redis_client()


async def close_redis() -> None:
    """关闭 Redis 连接池（应用关闭时调用）。"""
    await redis_client.aclose()


async def get_redis() -> AsyncGenerator[Redis, None]:
    """FastAPI 依赖：提供 Redis 客户端。"""
    yield redis_client
