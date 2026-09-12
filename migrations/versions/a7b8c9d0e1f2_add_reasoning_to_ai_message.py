"""add reasoning column to ai_message

Revision ID: a7b8c9d0e1f2
Revises: a1b2c3d4e5f7
Create Date: 2026-09-11 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a7b8c9d0e1f2'
down_revision: str | None = 'a1b2c3d4e5f7'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 模型推理内容（思维链，推理型模型才有；可空）
    op.add_column('ai_message', sa.Column('reasoning', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('ai_message', 'reasoning')
