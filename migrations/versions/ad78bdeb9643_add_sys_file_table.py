"""add sys_file table

Revision ID: ad78bdeb9643
Revises: c14c20ecdf47
Create Date: 2026-08-17 00:32:04.448379

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'ad78bdeb9643'
down_revision: str | None = 'c14c20ecdf47'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 通用文件元数据表：文件内容存存储后端，表中仅记录元数据与存储 key
    op.create_table(
        'sys_file',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('storage_key', sa.String(length=255), nullable=False),
        sa.Column('content_type', sa.String(length=127), nullable=False),
        sa.Column('size', sa.BigInteger(), nullable=False),
        sa.Column('ext', sa.String(length=32), nullable=True),
        sa.Column('sha256', sa.String(length=64), nullable=True),
        sa.Column('uploader_id', sa.Uuid(), nullable=True),
        sa.Column('biz_type', sa.String(length=32), nullable=True),
        sa.Column('status', sa.SmallInteger(), nullable=False),
        sa.Column('create_time', sa.DateTime(), nullable=False),
        sa.Column('update_time', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['uploader_id'], ['sys_user.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('storage_key'),
    )


def downgrade() -> None:
    op.drop_table('sys_file')
