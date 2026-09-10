"""Agent Service 层：业务规则、权限判断、流程编排。"""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import AgentStatus
from app.common.pagination import PageParams, PageResult
from app.core.exceptions import BizError, NotFoundError
from app.modules.agent.management.model import Agent, AgentVersion
from app.modules.agent.management.repository import AgentRepository, AgentVersionRepository
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


class AgentService:
    """Agent 管理服务（配置 + 版本 + 触发运行）。"""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
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
            # mode="json"：把 knowledge_ids 等 UUID 对象转成字符串，否则 JSON 列序列化报错
            config=req.config.model_dump(mode="json"),
            status=AgentStatus.DRAFT.value,
            current_version=0,
            creator_id=creator_id,
        )
        return AgentOut.model_validate(await self.repo.create(agent))

    # 更新 Agent（配置仅草稿可改；状态可直接切换）
    async def update(self, agent_id: UUID, req: AgentUpdate) -> AgentOut:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        # 非草稿状态仅允许切换状态，不允许改配置
        # if agent.status != AgentStatus.DRAFT.value:
        #     has_config_change = any(
        #         v is not None
        #         for v in (
        #             req.name,
        #             req.description,
        #             req.icon_id,
        #             req.model_id,
        #             req.organization_id,
        #             req.config,
        #         )
        #     )
        #     if has_config_change:
        #         raise BizError("仅草稿状态的 Agent 可修改配置，请先重新编辑")
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
            # mode="json"：把 knowledge_ids 等 UUID 对象转成字符串，否则 JSON 列序列化报错
            agent.config = req.config.model_dump(mode="json")
        if req.status is not None:
            agent.status = req.status.value
        return AgentOut.model_validate(await self.repo.update(agent))

    # 删除 Agent（仅草稿 / 停止状态可删除，避免误删运行中的线上服务）
    async def delete(self, agent_id: UUID) -> None:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        if agent.status not in (AgentStatus.DRAFT.value, AgentStatus.STOPPED.value):
            raise BizError("仅草稿或停止状态的 Agent 可删除")
        await self.repo.delete(agent)

    # 发布：将当前草稿配置生成 published 版本快照，更新 current_version，并进入运行中
    async def publish(self, agent_id: UUID, req: AgentPublish) -> AgentVersionOut:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        if agent.status != AgentStatus.DRAFT.value:
            raise BizError("仅草稿状态的 Agent 可发布")
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
        agent.status = AgentStatus.RUNNING.value
        await self.repo.update(agent)
        return AgentVersionOut.model_validate(created)

    # 版本列表
    async def list_versions(self, agent_id: UUID) -> list[AgentVersionOut]:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        versions = await self.version_repo.list_by_agent(agent_id)
        return [AgentVersionOut.model_validate(v) for v in versions]

    # 切换版本：将 current_version 指向指定历史版本快照（回滚）
    async def switch_version(self, agent_id: UUID, req: AgentVersionSwitch) -> AgentOut:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        if agent.current_version == 0:
            raise BizError("Agent 尚未发布，无法切换版本")
        target = await self.version_repo.get_by_version(agent_id, req.version)
        if not target:
            raise BizError(f"目标版本 {req.version} 不存在")
        if req.version == agent.current_version:
            raise BizError("目标版本已是当前版本")
        agent.current_version = req.version
        return AgentOut.model_validate(await self.repo.update(agent))

    # 触发运行：同步调用 AI 对话，返回模型回复（测试运行，不落库会话）
    async def run(self, agent_id: UUID, req: AgentRunRequest, user_id: UUID | None) -> AgentRunOut:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        if agent.status in (AgentStatus.PAUSED.value, AgentStatus.STOPPED.value):
            raise BizError("Agent 已暂停/停止，无法运行")
        if agent.current_version == 0:
            raise BizError("Agent 尚未发布，无法运行")
        # 延迟导入：避免 agent.management 与 ai 模块循环依赖
        from app.modules.ai.service import AiService

        # 不传 user_id：测试运行不落库会话，避免产生孤儿会话
        out = await AiService(self.db).chat(agent_id, req.input, None)
        return AgentRunOut(run_id="", status="success", reply=out.reply, model=out.model)
