"""Agent 域 API 聚合入口。

作为 agent 域的统一入口：
- 挂载 Agent 管理（management）与模型管理（model）子模块路由；
- 子模块 api 通过相对 prefix（/management /model）被本文件 include_router 聚合，
  最终 URL 形如 /api/v1/agent/management、/api/v1/agent/model。
"""

from fastapi import APIRouter

from app.modules.agent.management.api import router as management_router
from app.modules.agent.model.api import router as model_router

# Agent 域总路由（URL 前缀 /agent）
router = APIRouter(prefix="/agent", tags=["Agent"])

# 聚合子模块路由：/agent/management、/agent/model
router.include_router(management_router)
router.include_router(model_router)
