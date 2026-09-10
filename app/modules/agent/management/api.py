"""Agent API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。

本层禁止直接操作数据库或编写复杂业务逻辑。
路由风格：动作前置、id 放最后（如 /agent/publish/{id}）。
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageResult
from app.common.response import ApiResponse, success
from app.core.database import get_db
from app.middleware.authentication import CurrentUserDep, get_current_user
from app.middleware.permission import require_permissions
from app.modules.agent.management.codes import PermissionCode
from app.modules.agent.management.schema import (
    AgentCreate,
    AgentOut,
    AgentPublish,
    AgentQuery,
    AgentRunOut,
    AgentRunRequest,
    AgentUpdate,
    AgentVersionOut,
    AgentVersionSwitch,
)
from app.modules.agent.management.service import AgentService

router = APIRouter(prefix="/management", tags=["Agent-管理"], dependencies=[Depends(get_current_user)])

DbDep = Annotated[AsyncSession, Depends(get_db)]


# 依赖注入工厂：创建 AgentService 实例（绑定数据库会话）
def get_service(db: DbDep) -> AgentService:
    return AgentService(db)


# ---- 列表 / 创建（静态路径，注册在 /{agent_id} 之前）----
@router.post(
    "/list",
    response_model=ApiResponse[PageResult[AgentOut]],
    summary="Agent 列表",
    dependencies=[Depends(require_permissions(PermissionCode.AGENT_LIST))],
)
# Agent 列表：分页查询，支持关键字/状态等条件过滤
async def list_agents(
    query: AgentQuery,
    service: Annotated[AgentService, Depends(get_service)],
) -> ApiResponse[PageResult[AgentOut]]:
    return success(data=await service.list_page(query))


@router.post(
    "/create",
    response_model=ApiResponse[AgentOut],
    summary="创建 Agent",
    dependencies=[Depends(require_permissions(PermissionCode.AGENT_CREATE))],
)
# 创建 Agent：校验参数并入库，返回新创建的 Agent 信息
async def create_agent(
    req: AgentCreate,
    current: CurrentUserDep,
    service: Annotated[AgentService, Depends(get_service)],
) -> ApiResponse[AgentOut]:
    return success(data=await service.create(req, current.user_id), message="创建成功")


# ---- 详情 / 更新 / 删除（id 放最后）----
@router.get(
    "/{agent_id}",
    response_model=ApiResponse[AgentOut],
    summary="Agent 详情",
    dependencies=[Depends(require_permissions(PermissionCode.AGENT_LIST))],
)
# Agent 详情：按 ID 查询单个 Agent
async def get_agent(
    agent_id: UUID,
    service: Annotated[AgentService, Depends(get_service)],
) -> ApiResponse[AgentOut]:
    return success(data=await service.get(agent_id))


@router.put(
    "/{agent_id}",
    response_model=ApiResponse[AgentOut],
    summary="更新 Agent",
    dependencies=[Depends(require_permissions(PermissionCode.AGENT_UPDATE))],
)
# 更新 Agent：按 ID 修改 Agent 信息
async def update_agent(
    agent_id: UUID,
    req: AgentUpdate,
    service: Annotated[AgentService, Depends(get_service)],
) -> ApiResponse[AgentOut]:
    await service.update(agent_id, req)
    return success(message="更新成功")


@router.delete(
    "/{agent_id}",
    response_model=ApiResponse[None],
    summary="删除 Agent",
    dependencies=[Depends(require_permissions(PermissionCode.AGENT_DELETE))],
)
# 删除 Agent：按 ID 删除
async def delete_agent(
    agent_id: UUID,
    service: Annotated[AgentService, Depends(get_service)],
) -> ApiResponse[None]:
    await service.delete(agent_id)
    return success(message="删除成功")


# ---- 动作接口（动作前置、id 放最后）----
@router.post(
    "/publish/{agent_id}",
    response_model=ApiResponse[AgentVersionOut],
    summary="发布 Agent",
    dependencies=[Depends(require_permissions(PermissionCode.AGENT_PUBLISH))],
)
# 发布 Agent：为当前草稿生成新版本并设为线上版本
async def publish_agent(
    agent_id: UUID,
    req: AgentPublish,
    service: Annotated[AgentService, Depends(get_service)],
) -> ApiResponse[AgentVersionOut]:
    return success(data=await service.publish(agent_id, req), message="发布成功")


@router.get(
    "/versions/{agent_id}",
    response_model=ApiResponse[list[AgentVersionOut]],
    summary="Agent 版本列表",
    dependencies=[Depends(require_permissions(PermissionCode.AGENT_VERSION_LIST))],
)
# Agent 版本列表：按 ID 查询该 Agent 的全部历史版本
async def list_agent_versions(
    agent_id: UUID,
    service: Annotated[AgentService, Depends(get_service)],
) -> ApiResponse[list[AgentVersionOut]]:
    return success(data=await service.list_versions(agent_id))


@router.post(
    "/switch-version/{agent_id}",
    response_model=ApiResponse[AgentOut],
    summary="切换 Agent 版本（回滚）",
    dependencies=[Depends(require_permissions(PermissionCode.AGENT_VERSION_SWITCH))],
)
# 切换 Agent 版本：回滚到指定历史版本
async def switch_agent_version(
    agent_id: UUID,
    req: AgentVersionSwitch,
    service: Annotated[AgentService, Depends(get_service)],
) -> ApiResponse[AgentOut]:
    return success(data=await service.switch_version(agent_id, req), message="版本切换成功")


@router.post(
    "/run/{agent_id}",
    response_model=ApiResponse[AgentRunOut],
    summary="触发 Agent 运行",
    dependencies=[Depends(require_permissions(PermissionCode.AGENT_RUN))],
)
# 触发 Agent 运行：提交运行请求，返回运行任务信息
async def run_agent(
    agent_id: UUID,
    req: AgentRunRequest,
    current: CurrentUserDep,
    service: Annotated[AgentService, Depends(get_service)],
) -> ApiResponse[AgentRunOut]:
    return success(data=await service.run(agent_id, req, current.user_id), message="运行已触发")
