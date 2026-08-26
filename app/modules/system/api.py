"""System API 层：系统总入口。

作为 system 模块的统一入口：
- 挂载用户 / 角色 / 权限等管理子模块路由；
- 提供系统级健康检查与就绪探活接口。
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.response import ApiResponse, success
from app.core.database import get_db
from app.core.redis import get_redis
from app.modules.system.operation_log.api import router as operation_log_router
from app.modules.system.organization.api import router as organization_router
from app.modules.system.permission.api import router as permission_router
from app.modules.system.role.api import router as role_router
from app.modules.system.user.api import router as user_router

# 系统总路由（URL 前缀 /system）
router = APIRouter(prefix="/system", tags=["系统"])

# 聚合管理子模块路由：/system/user、/system/role、/system/permission、/system/operation-log、/system/organization
router.include_router(user_router)
router.include_router(role_router)
router.include_router(permission_router)
router.include_router(operation_log_router)
router.include_router(organization_router)

DbDep = Annotated[AsyncSession, Depends(get_db)]
RedisDep = Annotated[Redis, Depends(get_redis)]


class HealthOut(BaseModel):
    """健康检查响应（不依赖外部资源）。"""

    status: str = "ok"
    service: str = "business-backend"
    version: str = "0.1.0"


class ReadinessOut(BaseModel):
    """就绪检查响应（探测 MySQL / Redis 连通性）。"""

    status: str = "ok"
    checks: dict[str, bool]


@router.get("/health", response_model=ApiResponse[HealthOut], summary="健康检查（无外部依赖）")
async def health() -> ApiResponse[HealthOut]:
    """进程健康检查，供负载均衡 / 运维探针使用。"""
    return success(data=HealthOut())


@router.get(
    "/ready",
    response_model=ApiResponse[ReadinessOut],
    summary="就绪检查（依赖 MySQL / Redis）",
)
async def readiness(db: DbDep, redis: RedisDep) -> ApiResponse[ReadinessOut]:
    """就绪探活：验证 MySQL 与 Redis 连通性，用于验证基础设施接线。

    任一依赖不可用时整体状态标记为 `degraded` 而非直接报错。
    """
    checks: dict[str, bool] = {"database": False, "redis": False}

    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:
        checks["database"] = False

    try:
        await redis.ping()
        checks["redis"] = True
    except Exception:
        checks["redis"] = False

    status = "ok" if all(checks.values()) else "degraded"
    return success(data=ReadinessOut(status=status, checks=checks))
