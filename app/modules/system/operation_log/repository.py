"""系统操作日志 Repository 层：数据访问（写入 / 分页查询）。"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageParams, PageResult, paginate
from app.common.repository import BaseRepository
from app.modules.system.operation_log.model import OperationLog


class OperationLogRepository(BaseRepository):
    """操作日志数据访问仓库。"""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, OperationLog)

    # 写入一条操作日志并提交（追加式，不修改既有记录）
    async def create(self, log: OperationLog) -> OperationLog:
        self.db.add(log)
        await self.db.commit()
        return log

    # 分页查询操作日志，支持按模块/动作/操作人/状态筛选，按时间倒序
    async def list_page(
        self,
        params: PageParams,
        module: str | None = None,
        action: str | None = None,
        username: str | None = None,
        status: int | None = None,
    ) -> PageResult[Any]:
        stmt = select(OperationLog).order_by(OperationLog.create_time.desc())
        if module:
            stmt = stmt.where(OperationLog.module == module)
        if action:
            stmt = stmt.where(OperationLog.action == action)
        if username:
            stmt = stmt.where(OperationLog.username.like(f"%{username}%"))
        if status is not None:
            stmt = stmt.where(OperationLog.status == status)
        return await paginate(self.db, stmt, params)