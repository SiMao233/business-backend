"""add model / organization / agent tables

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-08-24 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: str | None = 'a1b2c3d4e5f6'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 模型供应商表
    op.create_table(
        'sys_model_provider',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('name', sa.String(length=64), nullable=False),
        sa.Column('code', sa.String(length=64), nullable=False),
        sa.Column('base_url', sa.String(length=255), nullable=True),
        sa.Column('api_key', sa.String(length=512), nullable=True),
        sa.Column('is_builtin', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('status', sa.SmallInteger(), nullable=False),
        sa.Column('create_time', sa.DateTime(), nullable=False),
        sa.Column('update_time', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('code'),
    )

    # 模型实例表
    op.create_table(
        'sys_model_instance',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('provider_id', sa.Uuid(), nullable=False),
        sa.Column('name', sa.String(length=64), nullable=False),
        sa.Column('code', sa.String(length=64), nullable=False),
        sa.Column('model_type', sa.String(length=16), nullable=False),
        sa.Column('max_tokens', sa.Integer(), nullable=True),
        sa.Column('status', sa.SmallInteger(), nullable=False),
        sa.Column('create_time', sa.DateTime(), nullable=False),
        sa.Column('update_time', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['provider_id'], ['sys_model_provider.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_sys_model_instance_provider_id', 'sys_model_instance', ['provider_id'])

    # 组织表（树状，parent_id 自引用）
    op.create_table(
        'sys_organization',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('parent_id', sa.Uuid(), nullable=True),
        sa.Column('name', sa.String(length=64), nullable=False),
        sa.Column('code', sa.String(length=64), nullable=False),
        sa.Column('description', sa.String(length=255), nullable=True),
        sa.Column('owner_id', sa.Uuid(), nullable=True),
        sa.Column('status', sa.SmallInteger(), nullable=False),
        sa.Column('create_time', sa.DateTime(), nullable=False),
        sa.Column('update_time', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['owner_id'], ['sys_user.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['parent_id'], ['sys_organization.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('code'),
    )
    op.create_index('ix_sys_organization_parent_id', 'sys_organization', ['parent_id'])

    # Agent 定义表
    op.create_table(
        'sys_agent',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('code', sa.String(length=64), nullable=False),
        sa.Column('description', sa.String(length=512), nullable=True),
        sa.Column('icon_id', sa.Uuid(), nullable=True),
        sa.Column('model_id', sa.Uuid(), nullable=True),
        sa.Column('organization_id', sa.Uuid(), nullable=True),
        sa.Column('config', sa.JSON(), nullable=False),
        sa.Column('status', sa.SmallInteger(), nullable=False),
        sa.Column('current_version', sa.Integer(), nullable=False),
        sa.Column('creator_id', sa.Uuid(), nullable=True),
        sa.Column('create_time', sa.DateTime(), nullable=False),
        sa.Column('update_time', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['creator_id'], ['sys_user.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['icon_id'], ['sys_file.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['model_id'], ['sys_model_instance.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['organization_id'], ['sys_organization.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('code'),
    )
    op.create_index('ix_sys_agent_organization_id', 'sys_agent', ['organization_id'])

    # Agent 版本表
    op.create_table(
        'sys_agent_version',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('agent_id', sa.Uuid(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('config', sa.JSON(), nullable=False),
        sa.Column('changelog', sa.String(length=512), nullable=True),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('create_time', sa.DateTime(), nullable=False),
        sa.Column('update_time', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['agent_id'], ['sys_agent.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_sys_agent_version_agent_id', 'sys_agent_version', ['agent_id'])


def downgrade() -> None:
    op.drop_index('ix_sys_agent_version_agent_id', table_name='sys_agent_version')
    op.drop_table('sys_agent_version')
    op.drop_index('ix_sys_agent_organization_id', table_name='sys_agent')
    op.drop_table('sys_agent')
    op.drop_index('ix_sys_organization_parent_id', table_name='sys_organization')
    op.drop_table('sys_organization')
    op.drop_index('ix_sys_model_instance_provider_id', table_name='sys_model_instance')
    op.drop_table('sys_model_instance')
    op.drop_table('sys_model_provider')