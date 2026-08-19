"""add is_builtin to sys_permission and sys_role

Revision ID: c14c20ecdf47
Revises: 08edb2815e3b
Create Date: 2026-08-16 03:23:57.508563

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c14c20ecdf47'
down_revision: str | None = '08edb2815e3b'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 系统内置标记：seed 创建的内置权限/角色置 True，界面创建的为 False
    op.add_column(
        'sys_permission',
        sa.Column('is_builtin', sa.Boolean(), nullable=False, server_default='0'),
    )
    op.add_column(
        'sys_role',
        sa.Column('is_builtin', sa.Boolean(), nullable=False, server_default='0'),
    )


def downgrade() -> None:
    op.drop_column('sys_role', 'is_builtin')
    op.drop_column('sys_permission', 'is_builtin')
