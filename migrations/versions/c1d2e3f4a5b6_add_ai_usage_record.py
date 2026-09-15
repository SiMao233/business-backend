"""add ai_usage_record table and model pricing columns

Revision ID: c1d2e3f4a5b6
Revises: b3c4d5e6f7a8
Create Date: 2026-09-13 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c1d2e3f4a5b6'
down_revision: str | None = 'b3c4d5e6f7a8'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 模型调用用量明细（追加写入，不修改；维度名称存快照，统计查询免 JOIN）
    op.create_table(
        'ai_usage_record',
        sa.Column('id', sa.Uuid(), nullable=False),
        # 统计维度（不建 FK：用户 / Agent 删除不应阻塞写入，靠快照追溯）
        sa.Column('user_id', sa.Uuid(), nullable=True),
        sa.Column('username', sa.String(length=50), nullable=True),
        sa.Column('agent_id', sa.Uuid(), nullable=True),
        sa.Column('agent_name', sa.String(length=255), nullable=True),
        sa.Column('organization_id', sa.Uuid(), nullable=True),
        sa.Column('model_instance_id', sa.Uuid(), nullable=True),
        sa.Column('provider_id', sa.Uuid(), nullable=True),
        sa.Column('provider_code', sa.String(length=64), nullable=True),
        sa.Column('model_code', sa.String(length=64), nullable=True),
        sa.Column('conversation_id', sa.Uuid(), nullable=True),
        sa.Column('message_id', sa.Uuid(), nullable=True),
        # 用量
        sa.Column('input_tokens', sa.Integer(), nullable=False),
        sa.Column('output_tokens', sa.Integer(), nullable=False),
        sa.Column('total_tokens', sa.Integer(), nullable=False),
        # 成本（单价快照，单位：元 / 千 token；改价不影响历史账）
        sa.Column('input_price', sa.Numeric(precision=18, scale=6), nullable=True),
        sa.Column('output_price', sa.Numeric(precision=18, scale=6), nullable=True),
        sa.Column('total_cost', sa.Numeric(precision=18, scale=6), nullable=False),
        # 上下文
        sa.Column('call_type', sa.String(length=16), nullable=False),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('status', sa.SmallInteger(), nullable=False),
        sa.Column('request_id', sa.String(length=64), nullable=True),
        # 业务自然日（写入时按 usage_timezone 预计算，供按天分组；避免运行时 CONVERT_TZ）
        sa.Column('usage_date', sa.Date(), nullable=False),
        sa.Column('create_time', sa.DateTime(), nullable=False),
        sa.Column('update_time', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_ai_usage_record_usage_date', 'ai_usage_record', ['usage_date'])
    op.create_index(
        'ix_ai_usage_record_usage_date_user_id', 'ai_usage_record', ['usage_date', 'user_id']
    )
    op.create_index(
        'ix_ai_usage_record_usage_date_agent_id', 'ai_usage_record', ['usage_date', 'agent_id']
    )
    op.create_index(
        'ix_ai_usage_record_usage_date_organization_id',
        'ai_usage_record',
        ['usage_date', 'organization_id'],
    )
    op.create_index(
        'ix_ai_usage_record_usage_date_model_instance_id',
        'ai_usage_record',
        ['usage_date', 'model_instance_id'],
    )
    op.create_index('ix_ai_usage_record_create_time', 'ai_usage_record', ['create_time'])

    # 模型实例单价（元 / 千 token；为空表示不计费）
    op.add_column(
        'sys_model_instance',
        sa.Column('input_price', sa.Numeric(precision=18, scale=6), nullable=True),
    )
    op.add_column(
        'sys_model_instance',
        sa.Column('output_price', sa.Numeric(precision=18, scale=6), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('sys_model_instance', 'output_price')
    op.drop_column('sys_model_instance', 'input_price')
    op.drop_index('ix_ai_usage_record_create_time', table_name='ai_usage_record')
    op.drop_index(
        'ix_ai_usage_record_usage_date_model_instance_id', table_name='ai_usage_record'
    )
    op.drop_index(
        'ix_ai_usage_record_usage_date_organization_id', table_name='ai_usage_record'
    )
    op.drop_index('ix_ai_usage_record_usage_date_agent_id', table_name='ai_usage_record')
    op.drop_index('ix_ai_usage_record_usage_date_user_id', table_name='ai_usage_record')
    op.drop_index('ix_ai_usage_record_usage_date', table_name='ai_usage_record')
    op.drop_table('ai_usage_record')
