"""系统操作日志 Service 层：写入审计日志与分页查询。"""

import json
from typing import Any
from uuid import UUID

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageParams, PageResult
from app.middleware.authentication import UserContext
from app.modules.system.operation_log.model import OperationLog
from app.modules.system.operation_log.repository import OperationLogRepository
from app.modules.system.operation_log.schema import OperationLogOut, OperationLogQuery


class OperationLogService:
    """操作日志服务：记录后台高风险操作（删除、分配权限等）并支持分页查询。"""

    def __init__(self, db: AsyncSession) -> None:
        self.repo = OperationLogRepository(db)

    # 记录一条操作日志（追加写入，独立提交；失败不影响主操作结果）
    async def record(
        self,
        *,
        user: UserContext | None,
        module: str,
        action: str,
        target_id: UUID | None = None,
        detail: dict[str, Any] | None = None,
        request: Request | None = None,
        status: int = 1,
        error_msg: str | None = None,
    ) -> None:
        log = OperationLog(
            user_id=user.user_id if user else None,
            username=user.username if user else None,
            module=module,
            action=action,
            target_id=target_id,
            method=request.method if request else None,
            path=request.url.path if request else None,
            ip=request.client.host if request and request.client else None,
            detail=json.dumps(detail, ensure_ascii=False) if detail else None,
            status=status,
            error_msg=error_msg,
        )
        await self.repo.create(log)

    # 分页查询操作日志
    async def list_page(self, query: OperationLogQuery) -> PageResult[OperationLogOut]:
        page_params = PageParams(page=query.page, pageSize=query.pageSize)
        page = await self.repo.list_page(
            page_params,
            module=query.module,
            action=query.action,
            username=query.username,
            status=query.status,
        )
        return PageResult(
            list=[OperationLogOut.model_validate(item) for item in page.list],
            total=page.total,
            page=page.page,
            pageSize=page.pageSize,
        )