"""Agent Service 层：业务规则、权限判断、流程编排。"""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import AgentStatus, AgentVersionStatus
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

# 状态转换白名单（通过 update.status 字段切换；发布走 publish 单独处理）
_STATUS_TRANSITIONS: dict[str, set[str]] = {
    AgentStatus.DRAFT.value: {AgentStatus.RUNNING.value},
    AgentStatus.RUNNING.value: {AgentStatus.PAUSED.value},  # 运行中只能暂停，不能直接停止
    AgentStatus.PAUSED.value: {AgentStatus.RUNNING.value, AgentStatus.STOPPED.value},
    AgentStatus.PENDING.value: {AgentStatus.STOPPED.value},
    AgentStatus.STOPPED.value: set(),  # 终态
}

# 可批量更新的简单字段（config/status 单独处理）
_SIMPLE_FIELDS = ("name", "description", "icon_id", "model_id", "organization_id")


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

    # 创建 Agent（草稿，current_version 为空串=未发布）
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
            current_version="",
            creator_id=creator_id,
        )
        return AgentOut.model_validate(await self.repo.create(agent))

    # 更新 Agent（改配置后进入待发布；draft 尚未发布过，保持 draft；stopped 为终态不可修改）
    # 返回 (更新后的 Agent, 变更前状态)：供 API 层记录状态流转日志
    async def update(self, agent_id: UUID, req: AgentUpdate) -> tuple[AgentOut, str]:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        # 停止是终态：不可修改配置、不可切换状态
        if agent.status == AgentStatus.STOPPED.value:
            raise BizError("Agent 已停止，不可修改")
        old_status = agent.status
        # 简单字段批量赋值（config/status 单独处理）
        for field in _SIMPLE_FIELDS:
            value = getattr(req, field)
            if value is not None:
                setattr(agent, field, value)
        if req.config is not None:
            # mode="json"：把 knowledge_ids 等 UUID 对象转成字符串，否则 JSON 列序列化报错
            agent.config = req.config.model_dump(mode="json")
            # 修改配置后进入待发布（draft 尚未发布过，保持 draft）
            if agent.status != AgentStatus.DRAFT.value:
                agent.status = AgentStatus.PENDING.value
        if req.status is not None:
            # 状态转换白名单校验（如：运行中不能直接停止，需先暂停）
            allowed = _STATUS_TRANSITIONS.get(agent.status, set())
            if req.status.value not in allowed:
                raise BizError(f"不允许从 {agent.status} 切换到 {req.status.value}")
            agent.status = req.status.value
        return AgentOut.model_validate(await self.repo.update(agent)), old_status

    # 删除 Agent（仅草稿 / 待发布 / 停止状态可删除，避免误删运行中的线上服务）
    # 返回被删除对象，供 API 层写入操作日志摘要
    async def delete(self, agent_id: UUID) -> Agent:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        if agent.status not in (
            AgentStatus.DRAFT.value,
            AgentStatus.STOPPED.value,
            AgentStatus.PENDING.value,
        ):
            raise BizError("仅草稿、待发布或停止状态的 Agent 可删除")
        await self.repo.delete(agent)
        return agent

    # 发布：draft/pending → running，用用户指定的版本号生成快照
    async def publish(self, agent_id: UUID, req: AgentPublish) -> AgentVersionOut:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        if agent.status not in (AgentStatus.DRAFT.value, AgentStatus.PENDING.value):
            raise BizError("仅草稿或待发布状态的 Agent 可发布")
        # 版本号唯一性校验（同 Agent 下）
        if await self.version_repo.get_by_version(agent_id, req.version):
            raise BizError(f"版本号 {req.version} 已存在")
        version = AgentVersion(
            agent_id=agent_id,
            version=req.version,
            config=dict(agent.config),
            changelog=req.changelog,
            status=AgentVersionStatus.PUBLISHED.value,
        )
        created = await self.version_repo.create(version)
        agent.current_version = req.version
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

    # 切换版本（回滚）：用目标版本快照覆盖当前配置，状态进入待发布（需再发布生效）
    # 返回 (更新后的 Agent, 回滚前版本号)：供 API 层记录版本回滚日志
    async def switch_version(
        self, agent_id: UUID, req: AgentVersionSwitch
    ) -> tuple[AgentOut, str]:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        # 停止是终态：不可回滚
        if agent.status == AgentStatus.STOPPED.value:
            raise BizError("Agent 已停止，不可回滚")
        if not agent.current_version:
            raise BizError("Agent 尚未发布，无法切换版本")
        target = await self.version_repo.get_by_version(agent_id, req.version)
        if not target:
            raise BizError(f"目标版本 {req.version} 不存在")
        if req.version == agent.current_version:
            raise BizError("目标版本已是当前版本")
        old_version = agent.current_version
        # 回滚：恢复配置到目标版本快照，进入待发布（需再发布生效）
        agent.config = dict(target.config)
        agent.current_version = req.version
        agent.status = AgentStatus.PENDING.value
        return AgentOut.model_validate(await self.repo.update(agent)), old_version

    # 触发运行：同步调用 AI 对话，返回模型回复（测试运行，不落库会话）
    async def run(self, agent_id: UUID, req: AgentRunRequest, user_id: UUID | None) -> AgentRunOut:
        agent = await self.repo.get_by_id(agent_id)
        if not agent:
            raise NotFoundError("Agent 不存在")
        if agent.status in (
            AgentStatus.PAUSED.value,
            AgentStatus.STOPPED.value,
            AgentStatus.PENDING.value,
        ):
            raise BizError("Agent 已暂停/停止/待发布，无法运行")
        if not agent.current_version:
            raise BizError("Agent 尚未发布，无法运行")
        # 延迟导入：避免 agent.management 与 ai 模块循环依赖
        from app.modules.ai.chat.service import AiChatService

        # 不传 user_id：测试运行不落库会话，避免产生孤儿会话
        out = await AiChatService(self.db).chat(agent_id, req.input, None)
        return AgentRunOut(run_id="", status="success", reply=out.reply, model=out.model)
