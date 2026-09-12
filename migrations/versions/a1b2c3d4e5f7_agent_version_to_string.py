"""agent version: int -> string (user-managed version numbers)

Revision ID: a1b2c3d4e5f7
Revises: f6a7b8c9d0e1
Create Date: 2026-09-11 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f7'
down_revision: str | None = 'f6a7b8c9d0e1'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # sys_agent.current_version: Integer → String(64)，存量数字自动转字符串
    op.alter_column(
        'sys_agent',
        'current_version',
        existing_type=sa.Integer(),
        type_=sa.String(length=64),
        existing_nullable=False,
        server_default='',
    )
    # sys_agent_version.version: Integer → String(64)
    op.alter_column(
        'sys_agent_version',
        'version',
        existing_type=sa.Integer(),
        type_=sa.String(length=64),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        'sys_agent',
        'current_version',
        existing_type=sa.String(length=64),
        type_=sa.Integer(),
        existing_nullable=False,
        server_default='0',
    )
    op.alter_column(
        'sys_agent_version',
        'version',
        existing_type=sa.String(length=64),
        type_=sa.Integer(),
        existing_nullable=False,
    )
