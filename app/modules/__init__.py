"""业务模块注册中心（Modular Monolith 模块聚合）。

新增业务模块步骤：
1. 在 `app/modules/` 下新建模块目录（含 api / service / repository / model / schema 分层）
2. 在模块 `api.py` 中定义 `APIRouter`
3. 在本文件 `MODULES` 列表中加入该 router，应用启动时自动挂载到 `/api/v1` 前缀下
"""

from fastapi import APIRouter

from app.modules.agent.api import router as agent_router
from app.modules.ai.api import router as ai_router
from app.modules.file.api import router as file_router
from app.modules.iam.api import router as iam_router
from app.modules.knowledge.api import router as knowledge_router
from app.modules.system.api import router as system_router

# 按模块注册；后续模块若存在依赖关系，可按依赖顺序调整
# 说明：organization 已并入 system 子模块，model 已并入 agent 子模块，由各自聚合入口统一挂载
MODULES: list[APIRouter] = [
    iam_router,
    file_router,
    knowledge_router,
    agent_router,
    ai_router,
    system_router,
]


def build_api_router() -> APIRouter:
    """聚合所有业务模块路由，返回统一 api_router。"""
    api_router = APIRouter()
    for module_router in MODULES:
        api_router.include_router(module_router)
    return api_router
