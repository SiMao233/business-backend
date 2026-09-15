"""用量统计 Repository 层：明细写入与多维聚合查询。"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Select, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import UsageDimension
from app.common.pagination import PageParams, PageResult, paginate
from app.common.repository import BaseRepository
from app.modules.ai.model import AiConversation, AiMessage
from app.modules.knowledge.model import KnowledgeChunk, KnowledgeDocument
from app.modules.system.organization.model import Organization
from app.modules.usage.model import AiUsageRecord

# 聚合公共列：调用次数 / token / 成本 / 失败次数
_COUNT_COL = func.count().label("calls")
_INPUT_TOKENS_COL = func.coalesce(func.sum(AiUsageRecord.input_tokens), 0).label("input_tokens")
_OUTPUT_TOKENS_COL = func.coalesce(func.sum(AiUsageRecord.output_tokens), 0).label("output_tokens")
_TOTAL_TOKENS_COL = func.coalesce(func.sum(AiUsageRecord.total_tokens), 0).label("total_tokens")
_TOTAL_COST_COL = func.coalesce(func.sum(AiUsageRecord.total_cost), 0).label("total_cost")
_ERROR_COUNT_COL = func.coalesce(
    func.sum(case((AiUsageRecord.status == 0, 1), else_=0)), 0
).label("error_count")

# 维度 → (分组 ID 列, 展示名称列)
# organization 无名称快照，查询时 outerjoin sys_organization 取当前名称（被删组织名称为 NULL）。
_DIMENSION_COLUMNS: dict[str, tuple[Any, Any]] = {
    UsageDimension.USER.value: (AiUsageRecord.user_id, AiUsageRecord.username),
    UsageDimension.AGENT.value: (AiUsageRecord.agent_id, AiUsageRecord.agent_name),
    UsageDimension.MODEL.value: (AiUsageRecord.model_instance_id, AiUsageRecord.model_code),
    UsageDimension.PROVIDER.value: (AiUsageRecord.provider_id, AiUsageRecord.provider_code),
    UsageDimension.ORGANIZATION.value: (AiUsageRecord.organization_id, Organization.name),
}


class UsageRepository(BaseRepository):
    """用量明细数据访问仓库。"""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, AiUsageRecord)

    # ---- 写入 ----
    async def create(self, record: AiUsageRecord) -> AiUsageRecord:
        """追加写入一条用量明细并提交。"""
        self.db.add(record)
        await self.db.commit()
        return record

    # ---- 明细分页 ----
    async def list_page(
        self,
        params: PageParams,
        *,
        start: datetime,
        end: datetime,
        scope_user_id: UUID | None = None,
        user_id: UUID | None = None,
        agent_id: UUID | None = None,
        model_instance_id: UUID | None = None,
        organization_id: UUID | None = None,
        call_type: str | None = None,
        status: int | None = None,
    ) -> PageResult[Any]:
        stmt: Select = (
            select(AiUsageRecord)
            .where(AiUsageRecord.create_time >= start, AiUsageRecord.create_time < end)
            .order_by(AiUsageRecord.create_time.desc())
        )
        # 数据范围（普通用户仅自己）
        if scope_user_id is not None:
            stmt = stmt.where(AiUsageRecord.user_id == scope_user_id)
        if user_id is not None:
            stmt = stmt.where(AiUsageRecord.user_id == user_id)
        if agent_id is not None:
            stmt = stmt.where(AiUsageRecord.agent_id == agent_id)
        if model_instance_id is not None:
            stmt = stmt.where(AiUsageRecord.model_instance_id == model_instance_id)
        if organization_id is not None:
            stmt = stmt.where(AiUsageRecord.organization_id == organization_id)
        if call_type is not None:
            stmt = stmt.where(AiUsageRecord.call_type == call_type)
        if status is not None:
            stmt = stmt.where(AiUsageRecord.status == status)
        return await paginate(self.db, stmt, params)

    # ---- 汇总（按维度分组）----
    async def summary_by(
        self,
        dimension: str,
        *,
        start: datetime,
        end: datetime,
        scope_user_id: UUID | None = None,
        top: int = 20,
    ) -> list[dict[str, Any]]:
        id_col, name_col = _DIMENSION_COLUMNS[dimension]
        stmt: Select = select(
            id_col.label("dimension_id"),
            name_col.label("dimension_name"),
            _COUNT_COL,
            _INPUT_TOKENS_COL,
            _OUTPUT_TOKENS_COL,
            _TOTAL_TOKENS_COL,
            _TOTAL_COST_COL,
            _ERROR_COUNT_COL,
        ).where(AiUsageRecord.create_time >= start, AiUsageRecord.create_time < end)
        # 组织维度需 join 组织表取名称
        if dimension == UsageDimension.ORGANIZATION.value:
            stmt = stmt.outerjoin(Organization, Organization.id == AiUsageRecord.organization_id)
        if scope_user_id is not None:
            stmt = stmt.where(AiUsageRecord.user_id == scope_user_id)
        stmt = (
            stmt.group_by(id_col, name_col)
            .order_by(func.coalesce(func.sum(AiUsageRecord.total_tokens), 0).desc())
            .limit(top)
        )
        rows = (await self.db.execute(stmt)).all()
        return [dict(row._mapping) for row in rows]

    async def summary_total(
        self, *, start: datetime, end: datetime, scope_user_id: UUID | None = None
    ) -> dict[str, Any]:
        """汇总合计（不分组）。"""
        stmt: Select = select(
            _COUNT_COL,
            _INPUT_TOKENS_COL,
            _OUTPUT_TOKENS_COL,
            _TOTAL_TOKENS_COL,
            _TOTAL_COST_COL,
            _ERROR_COUNT_COL,
        ).where(AiUsageRecord.create_time >= start, AiUsageRecord.create_time < end)
        if scope_user_id is not None:
            stmt = stmt.where(AiUsageRecord.user_id == scope_user_id)
        row = (await self.db.execute(stmt)).one()
        return dict(row._mapping)

    # ---- 趋势（按业务自然日）----
    async def trend_by_day(
        self, *, start: datetime, end: datetime, scope_user_id: UUID | None = None
    ) -> list[dict[str, Any]]:
        stmt: Select = select(
            AiUsageRecord.usage_date.label("stat_date"),
            _COUNT_COL,
            _INPUT_TOKENS_COL,
            _OUTPUT_TOKENS_COL,
            _TOTAL_TOKENS_COL,
            _TOTAL_COST_COL,
        ).where(AiUsageRecord.create_time >= start, AiUsageRecord.create_time < end)
        if scope_user_id is not None:
            stmt = stmt.where(AiUsageRecord.user_id == scope_user_id)
        stmt = stmt.group_by(AiUsageRecord.usage_date).order_by(AiUsageRecord.usage_date.asc())
        rows = (await self.db.execute(stmt)).all()
        return [dict(row._mapping) for row in rows]

    # ---- AI 用量概览 ----
    async def overview_ai(
        self, *, start: datetime, end: datetime, scope_user_id: UUID | None = None
    ) -> dict[str, Any]:
        stmt: Select = select(
            _COUNT_COL,
            _INPUT_TOKENS_COL,
            _OUTPUT_TOKENS_COL,
            _TOTAL_TOKENS_COL,
            _TOTAL_COST_COL,
            func.avg(AiUsageRecord.latency_ms).label("avg_latency_ms"),
            _ERROR_COUNT_COL,
        ).where(AiUsageRecord.create_time >= start, AiUsageRecord.create_time < end)
        if scope_user_id is not None:
            stmt = stmt.where(AiUsageRecord.user_id == scope_user_id)
        row = (await self.db.execute(stmt)).one()
        return dict(row._mapping)

    # ---- 业务用量计数（实时统计，不落汇总表）----
    async def biz_counts(
        self, *, start: datetime, end: datetime, scope_user_id: UUID | None = None
    ) -> dict[str, int]:
        conv_stmt = select(func.count()).select_from(AiConversation).where(
            AiConversation.create_time >= start, AiConversation.create_time < end
        )
        if scope_user_id is not None:
            conv_stmt = conv_stmt.where(AiConversation.user_id == scope_user_id)

        msg_stmt = (
            select(func.count())
            .select_from(AiMessage)
            .join(AiConversation, AiConversation.id == AiMessage.conversation_id)
            .where(AiMessage.create_time >= start, AiMessage.create_time < end)
        )
        if scope_user_id is not None:
            msg_stmt = msg_stmt.where(AiConversation.user_id == scope_user_id)

        doc_stmt = select(func.count()).select_from(KnowledgeDocument).where(
            KnowledgeDocument.create_time >= start, KnowledgeDocument.create_time < end
        )
        if scope_user_id is not None:
            doc_stmt = doc_stmt.where(KnowledgeDocument.uploader_id == scope_user_id)

        chunk_stmt = select(func.count()).select_from(KnowledgeChunk).where(
            KnowledgeChunk.create_time >= start, KnowledgeChunk.create_time < end
        )
        if scope_user_id is not None:
            # chunk 无 uploader 列，经文档关联到上传者，保持与 documents 口径一致
            chunk_stmt = chunk_stmt.join(
                KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunk.document_id
            ).where(KnowledgeDocument.uploader_id == scope_user_id)

        return {
            "conversations": (await self.db.execute(conv_stmt)).scalar_one(),
            "messages": (await self.db.execute(msg_stmt)).scalar_one(),
            "documents": (await self.db.execute(doc_stmt)).scalar_one(),
            "chunks": (await self.db.execute(chunk_stmt)).scalar_one(),
        }
