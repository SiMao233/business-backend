"""Agent Repository 层：数据访问（CRUD / 查询）。"""

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import AgentStatus
from app.common.pagination import PageParams, PageResult, paginate
from app.common.repository import BaseRepository
from app.modules.agent.management.model import Agent, AgentVersion


class AgentRepository(BaseRepository):
    """Agent 定义数据访问仓库。"""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, Agent)

    # 根据主键 ID 查询 Agent
    async def get_by_id(self, agent_id: UUID) -> Agent | None:
        stmt = select(Agent).where(Agent.id == agent_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 根据业务编码查询（唯一性校验，可排除自身）
    async def get_by_code(self, code: str, exclude_id: UUID | None = None) -> Agent | None:
        stmt = select(Agent).where(Agent.code == code)
        if exclude_id is not None:
            stmt = stmt.where(Agent.id != exclude_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 分页查询 Agent 列表，支持关键字 / 状态 / 组织筛选
    async def list_page(
        self,
        params: PageParams,
        keyword: str | None = None,
        status: AgentStatus | None = None,
        organization_id: UUID | None = None,
    ) -> PageResult[Any]:
        stmt = select(Agent).order_by(Agent.id.desc())
        if keyword:
            like = f"%{keyword}%"
            stmt = stmt.where(Agent.name.like(like) | Agent.code.like(like))
        if status is not None:
            stmt = stmt.where(Agent.status == status.value)
        if organization_id is not None:
            stmt = stmt.where(Agent.organization_id == organization_id)
        return await paginate(self.db, stmt, params)

    # 新增 Agent：写入并提交，刷新后返回对象
    async def create(self, agent: Agent) -> Agent:
        self.db.add(agent)
        await self.db.commit()
        stmt = select(Agent).where(Agent.id == agent.id)
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 更新 Agent：提交变更并刷新，返回更新后的对象
    async def update(self, agent: Agent) -> Agent:
        await self.db.commit()
        stmt = select(Agent).where(Agent.id == agent.id)
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 删除 Agent：从数据库移除并提交事务
    async def delete(self, agent: Agent) -> None:
        await self.db.delete(agent)
        await self.db.commit()


class AgentVersionRepository(BaseRepository):
    """Agent 版本数据访问仓库。"""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, AgentVersion)

    # 查询某 Agent 的全部版本（按创建时间倒序，字符串版本号无法按数字排序）
    async def list_by_agent(self, agent_id: UUID) -> list[AgentVersion]:
        stmt = (
            select(AgentVersion)
            .where(AgentVersion.agent_id == agent_id)
            .order_by(AgentVersion.create_time.desc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # 查询某 Agent 的最新版本（按创建时间倒序）
    async def get_latest(self, agent_id: UUID) -> AgentVersion | None:
        stmt = (
            select(AgentVersion)
            .where(AgentVersion.agent_id == agent_id)
            .order_by(AgentVersion.create_time.desc())
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 查询指定版本号（运行时读取线上配置快照用）
    async def get_by_version(self, agent_id: UUID, version: str) -> AgentVersion | None:
        stmt = select(AgentVersion).where(
            AgentVersion.agent_id == agent_id, AgentVersion.version == version
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 新增版本：写入并提交，刷新后返回对象
    async def create(self, version: AgentVersion) -> AgentVersion:
        self.db.add(version)
        await self.db.commit()
        stmt = select(AgentVersion).where(AgentVersion.id == version.id)
        result = await self.db.execute(stmt)
        return result.scalar_one()
