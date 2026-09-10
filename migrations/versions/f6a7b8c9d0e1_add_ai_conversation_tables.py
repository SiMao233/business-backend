"""add ai conversation tables (ai_conversation / ai_message)

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-09-06 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f6a7b8c9d0e1'
down_revision: str | None = 'e5f6a7b8c9d0'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # AI 对话会话表
    op.create_table(
        'ai_conversation',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('agent_id', sa.Uuid(), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=True),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('create_time', sa.DateTime(), nullable=False),
        sa.Column('update_time', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['agent_id'], ['sys_agent.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['sys_user.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_ai_conversation_agent_id', 'ai_conversation', ['agent_id'])
    op.create_index('ix_ai_conversation_user_id', 'ai_conversation', ['user_id'])

    # 会话内消息表
    op.create_table(
        'ai_message',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('conversation_id', sa.Uuid(), nullable=False),
        sa.Column('role', sa.String(length=16), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('create_time', sa.DateTime(), nullable=False),
        sa.Column('update_time', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['conversation_id'], ['ai_conversation.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_ai_message_conversation_id', 'ai_message', ['conversation_id'])


def downgrade() -> None:
    op.drop_index('ix_ai_message_conversation_id', table_name='ai_message')
    op.drop_table('ai_message')

    op.drop_index('ix_ai_conversation_user_id', table_name='ai_conversation')
    op.drop_index('ix_ai_conversation_agent_id', table_name='ai_conversation')
    op.drop_table('ai_conversation')
