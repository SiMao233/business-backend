"""agent status: smallint -> string enum (draft/running/paused/stopped)

Revision ID: d4e5f6a7b8c9
Revises: b2c3d4e5f6a7
Create Date: 2026-08-28 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: str | None = 'b2c3d4e5f6a7'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 先改列类型：SmallInteger → String(16)，默认 draft
    # （必须先 ALTER 再 UPDATE，否则往整型列写字符串会报 MySQL 1366）
    op.alter_column(
        'sys_agent',
        'status',
        existing_type=sa.SmallInteger(),
        type_=sa.String(length=16),
        existing_nullable=False,
        server_default='draft',
    )
    # 再映射存量数据：1（启用）→ running，0（禁用）→ stopped
    op.execute("UPDATE sys_agent SET status = 'running' WHERE status = '1'")
    op.execute("UPDATE sys_agent SET status = 'stopped' WHERE status = '0'")


def downgrade() -> None:
    # 先映射回整数：running → 1，其余（draft/paused/stopped）→ 0
    op.execute("UPDATE sys_agent SET status = '1' WHERE status = 'running'")
    op.execute("UPDATE sys_agent SET status = '0' WHERE status IN ('draft', 'paused', 'stopped')")
    # 再改回列类型：String(16) → SmallInteger，默认 1
    op.alter_column(
        'sys_agent',
        'status',
        existing_type=sa.String(length=16),
        type_=sa.SmallInteger(),
        existing_nullable=False,
        server_default='1',
    )