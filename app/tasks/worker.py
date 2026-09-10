"""ARQ 后台任务 Worker 入口。

启动方式（项目根目录执行）：:

    uv run arq app.tasks.worker.WorkerSettings

任务队列基于项目已有的 Redis（配置见 app/core/config.py 的 redis_*）。
当前注册的任务：
- `run_agent_chat`：异步执行 Agent 对话（由 app/modules/ai/ 入队，长耗时调用走后台）；
- `process_document`：后台处理知识库文档（解析 → 切分 → embedding → 写 Qdrant → 双写 MySQL）。
"""

from arq import create_pool
from arq.connections import RedisSettings

from app.core.config import get_settings


def _redis_settings() -> RedisSettings:
    """从应用配置构造 ARQ Redis 连接参数。"""
    settings = get_settings()
    return RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        database=settings.redis_db,
        password=settings.redis_password,
    )


async def enqueue_job(function: str, *args) -> None:
    """入队 ARQ 任务（每次新建连接，入队后关闭）。"""
    redis = await create_pool(_redis_settings())
    try:
        await redis.enqueue_job(function, *args)
    finally:
        await redis.aclose()


async def run_agent_chat(ctx: dict, agent_id: str, user_input: str, user_id: str | None = None) -> dict:
    """后台执行 Agent 对话（ARQ 任务）。

    由 `app/modules/ai/service.py` 入队调用；此处延迟导入 AiService，
    避免 worker 启动时与 FastAPI 应用模块产生循环依赖。
    """
    from app.modules.ai.service import AiService

    return await AiService.chat_task(agent_id, user_input, user_id)


async def process_document(ctx: dict, document_id: str) -> dict:
    """后台处理知识库文档（ARQ 任务）。

    由 `app/modules/knowledge/service.py` 入队调用；延迟导入避免循环依赖。
    """
    from app.modules.knowledge.service import process_document_task

    return await process_document_task(document_id)


class WorkerSettings:
    """ARQ Worker 配置。"""

    functions = [run_agent_chat, process_document]
    redis_settings = _redis_settings()
    # 并发任务数
    max_jobs = 10
    # 任务超时（秒）
    job_timeout = 300
    # 定时任务（预留，后续文档切分/索引等）
    cron_jobs: list = []
