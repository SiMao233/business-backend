"""分页通用模型与工具。"""

from typing import Annotated, Any

from fastapi import Query
from pydantic import BaseModel, Field
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession


class PageParams(BaseModel):
    """分页参数。"""

    page: int = Field(default=1, ge=1, description="页码（从 1 开始）")
    pageSize: int = Field(default=20, ge=1, le=100, description="每页条数（最大 100）")


async def get_page_params(
    page: Annotated[int, Query(ge=1)] = 1,
    pageSize: Annotated[int, Query(ge=10, le=100)] = 20,
) -> PageParams:
    """FastAPI 依赖：从查询参数解析分页参数。"""
    return PageParams(page=page, pageSize=pageSize)


class PageResult[T](BaseModel):
    """分页结果。"""

    list: list[T]
    total: int
    page: int
    pageSize: int


async def paginate(db: AsyncSession, stmt: Select, params: PageParams) -> PageResult[Any]:
    """执行分页查询：返回总条数 `total` 与当前页 `list`。"""
    count_stmt = select(func.count()).select_from(stmt.order_by(None).subquery())
    total = (await db.execute(count_stmt)).scalar_one()
    rows = await db.execute(
        stmt.offset((params.page - 1) * params.pageSize).limit(params.pageSize)
    )
    items = list(rows.scalars().all())
    return PageResult(list=items, total=total, page=params.page, pageSize=params.pageSize)
