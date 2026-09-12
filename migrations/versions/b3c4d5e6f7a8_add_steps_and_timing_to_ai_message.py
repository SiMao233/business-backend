"""add steps and timing columns to ai_message

Revision ID: b3c4d5e6f7a8
Revises: a7b8c9d0e1f2
Create Date: 2026-09-13 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b3c4d5e6f7a8'
down_revision: str | None = 'a7b8c9d0e1f2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 工具 / 检索步骤（历史回看用；无步骤时为 NULL）
    op.add_column('ai_message', sa.Column('steps', sa.JSON(), nullable=True))
    # 思考耗时（毫秒）：首个 reasoning → 首个正文；无推理内容时为 NULL
    op.add_column('ai_message', sa.Column('thinking_ms', sa.Integer(), nullable=True))
    # 首字节耗时（毫秒）：服务端进入预检 → 首个正文；未产出正文时为 NULL
    op.add_column('ai_message', sa.Column('ttft_ms', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('ai_message', 'ttft_ms')
    op.drop_column('ai_message', 'thinking_ms')
    op.drop_column('ai_message', 'steps')
