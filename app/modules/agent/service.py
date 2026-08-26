"""Agent Service 层：业务规则、权限判断、流程编排。"""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageParams, PageResult
from app.core.exceptions import BizError, NotFoundError
from app.modules.agent.model import Agent, AgentVersion
from app.modules.agent.repository import AgentRepository, AgentVersionRepository
from app.modules.agent.schema import (
    AgentCreate,
    AgentOut,
    AgentPublish,
    AgentQuery,
    AgentRunOut,
    AgentRunRequest,
    AgentUpdate,
    AgentVersionOut,
)


class AgentService:
    """Agent 管理服务（配置 + 版本 + 触发运行）。"""

    def __init__(self, db: AsyncSession) -> None:
        self.repo = AgentRepository(db)
        self.version_repo = AgentVersionRepository(db)

    # 分页查询 Agent 列表
    async def list_page(self, query: AgentQuery) -> PageResult[AgentOut]:
        page_params = PageParams(page=query.page, pageSize=query.pageSize)
        page = await self.repo.list_page(
            page_params,
            keyword=query.keyword,
            status=query.status,
            organization_id=query.organization_id,
        )
        return PageResult(
            list=[AgentOut.model_validate(item) for item in page.list],
            total=page.total,
            page=page.page,
            pageSize=page.pageSize,
        )

    # 查询单个 Agent 详情
    async def get(self, agent_id: UUID) -> AgentOut:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        return AgentOut.model_validate(agent)

    # 创建 Agent（草稿，current_version=0）
    async def create(self, req: AgentCreate, creator_id: UUID | None) -> AgentOut:
        if await self.repo.get_by_code(req.code):
            raise BizError("Agent 编码已存在")
        agent = Agent(
            name=req.name,
            code=req.code,
            description=req.description,
            icon_id=req.icon_id,
            model_id=req.model_id,
            organization_id=req.organization_id,
            config=req.config.model_dump(),
            status=req.status,
            current_version=0,
            creator_id=creator_id,
        )
        return AgentOut.model_validate(await self.repo.create(agent))

    # 更新 Agent（未发布直改；已发布则修改当前配置，下次发布生成新版本）
    async def update(self, agent_id: UUID, req: AgentUpdate) -> AgentOut:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        if req.name is not None:
            agent.name = req.name
        if req.description is not None:
            agent.description = req.description
        if req.icon_id is not None:
            agent.icon_id = req.icon_id
        if req.model_id is not None:
            agent.model_id = req.model_id
        if req.organization_id is not None:
            agent.organization_id = req.organization_id
        if req.config is not None:
            agent.config = req.config.model_dump()
        if req.status is not None:
            agent.status = req.status
        return AgentOut.model_validate(await self.repo.update(agent))

    # 删除 Agent
    async def delete(self, agent_id: UUID) -> None:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        await self.repo.delete(agent)

    # 发布：将当前配置生成 published 版本快照，并更新 current_version
    async def publish(self, agent_id: UUID, req: AgentPublish) -> AgentVersionOut:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        latest = await self.version_repo.get_latest(agent_id)
        next_version = (latest.version if latest else 0) + 1
        version = AgentVersion(
            agent_id=agent_id,
            version=next_version,
            config=dict(agent.config),
            changelog=req.changelog,
            status="published",
        )
        created = await self.version_repo.create(version)
        agent.current_version = next_version
        await self.repo.update(agent)
        return AgentVersionOut.model_validate(created)

    # 版本列表
    async def list_versions(self, agent_id: UUID) -> list[AgentVersionOut]:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        versions = await self.version_repo.list_by_agent(agent_id)
        return [AgentVersionOut.model_validate(v) for v in versions]

    # 触发运行：本版仅返回受理回执（ai-service 尚未接入）
    async def run(self, agent_id: UUID, req: AgentRunRequest, user_id: UUID | None) -> AgentRunOut:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        if agent.status != 1:
            raise BizError("Agent 已停用，无法运行")
        if agent.current_version == 0:
            raise BizError("Agent 尚未发布，无法运行")
        # TODO(ai-service): 调用独立 ai-service 触发运行，返回真实 run_id
        return AgentRunOut(run_id="", status="running")
