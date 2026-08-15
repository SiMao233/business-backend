"""应用入口：创建并暴露 FastAPI 应用实例。

启动方式（项目根目录执行）：:

    uv run uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.common.response import ApiResponse, success
from app.core.config import get_settings
from app.core.database import dispose_engine
from app.core.exceptions import register_exception_handlers
from app.core.redis import close_redis

# 导入所有 ORM 模型，确保它们注册到 SQLAlchemy registry（否则字符串关系如 "User" 解析不到）
import app.models  # noqa: F401

from app.modules import build_api_router

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化资源，关闭时释放连接池。

    数据库引擎与 Redis 客户端已在模块导入时惰性创建，此处仅在退出时统一释放。
    """
    yield
    await dispose_engine()
    await close_redis()


def create_app() -> FastAPI:
    """应用工厂：组装 CORS、全局异常处理、业务路由与探针。"""
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        debug=settings.app_debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url=None if settings.app_env != "production" else None,
    )

    # CORS（来源列表由配置控制）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 全局异常处理（统一输出 ApiResponse 结构）
    register_exception_handlers(app)

    # 挂载业务模块路由（/api/v1 前缀）
    app.include_router(build_api_router(), prefix=settings.app_api_prefix)

    # 存活探针（无外部依赖，供 Docker / 负载均衡使用）
    @app.get("/health", response_model=ApiResponse[dict], summary="存活探针", include_in_schema=False)
    async def health() -> ApiResponse[dict]:
        return success(data={"status": "ok"})

    return app


# 供 `uvicorn app.main:app` 直接加载
app = create_app()
