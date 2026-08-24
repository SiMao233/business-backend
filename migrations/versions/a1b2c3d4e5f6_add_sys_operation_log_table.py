"""add sys_operation_log table

Revision ID: a1b2c3d4e5f6
Revises: ad78bdeb9643
Create Date: 2026-08-20 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: str | None = 'ad78bdeb9643'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 操作日志表：记录后台高风险操作（删除、分配权限等）用于审计追溯
    op.create_table(
        'sys_operation_log',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=True),
        sa.Column('username', sa.String(length=50), nullable=True),
        sa.Column('module', sa.String(length=50), nullable=False),
        sa.Column('action', sa.String(length=50), nullable=False),
        sa.Column('target_id', sa.Uuid(), nullable=True),
        sa.Column('method', sa.String(length=10), nullable=True),
        sa.Column('path', sa.String(length=255), nullable=True),
        sa.Column('ip', sa.String(length=50), nullable=True),
        sa.Column('detail', sa.Text(), nullable=True),
        sa.Column('status', sa.SmallInteger(), nullable=False),
        sa.Column('error_msg', sa.String(length=500), nullable=True),
        sa.Column('create_time', sa.DateTime(), nullable=False),
        sa.Column('update_time', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['sys_user.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_sys_operation_log_create_time', 'sys_operation_log', ['create_time'])
    op.create_index('ix_sys_operation_log_user_id', 'sys_operation_log', ['user_id'])
    op.create_index('ix_sys_operation_log_module', 'sys_operation_log', ['module'])


def downgrade() -> None:
    op.drop_index('ix_sys_operation_log_module', table_name='sys_operation_log')
    op.drop_index('ix_sys_operation_log_user_id', table_name='sys_operation_log')
    op.drop_index('ix_sys_operation_log_create_time', table_name='sys_operation_log')
    op.drop_table('sys_operation_log')