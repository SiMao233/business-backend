"""AI 能力 Repository 层：数据访问（会话 / 消息）。"""

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageParams, PageResult, paginate
from app.common.repository import BaseRepository
from app.modules.ai.model import AiConversation, AiMessage


class ConversationRepository(BaseRepository):
    """AI 对话会话数据访问仓库。"""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, AiConversation)

    # 根据主键 ID 查询会话
    async def get_by_id(self, conversation_id: UUID) -> AiConversation | None:
        stmt = select(AiConversation).where(AiConversation.id == conversation_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 分页查询会话列表（按用户 + 可选 Agent 筛选，创建时间倒序）
    async def list_page(
        self,
        params: PageParams,
        user_id: UUID,
        agent_id: UUID | None = None,
    ) -> PageResult[Any]:
        stmt = (
            select(AiConversation)
            .where(AiConversation.user_id == user_id)
            .order_by(AiConversation.update_time.desc())
        )
        if agent_id is not None:
            stmt = stmt.where(AiConversation.agent_id == agent_id)
        return await paginate(self.db, stmt, params)

    # 新增会话：写入并提交，刷新后返回对象
    async def create(self, conv: AiConversation) -> AiConversation:
        self.db.add(conv)
        await self.db.commit()
        stmt = select(AiConversation).where(AiConversation.id == conv.id)
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 更新会话：提交变更并刷新，返回更新后的对象
    async def update(self, conv: AiConversation) -> AiConversation:
        await self.db.commit()
        stmt = select(AiConversation).where(AiConversation.id == conv.id)
        result = await self.db.execute(stmt)
        return result.scalar_one()

    # 删除会话：从数据库移除并提交事务（消息由 DB CASCADE 删除）
    async def delete(self, conv: AiConversation) -> None:
        await self.db.delete(conv)
        await self.db.commit()


class MessageRepository(BaseRepository):
    """AI 会话消息数据访问仓库。"""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, AiMessage)

    # 查询某会话的全部消息（按创建时间升序，用于多轮上下文组装）
    async def list_by_conversation(self, conversation_id: UUID) -> list[AiMessage]:
        stmt = (
            select(AiMessage)
            .where(AiMessage.conversation_id == conversation_id)
            .order_by(AiMessage.create_time.asc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # 新增消息：写入并提交，刷新后返回对象
    async def create(self, msg: AiMessage) -> AiMessage:
        self.db.add(msg)
        await self.db.commit()
        stmt = select(AiMessage).where(AiMessage.id == msg.id)
        result = await self.db.execute(stmt)
        return result.scalar_one()
