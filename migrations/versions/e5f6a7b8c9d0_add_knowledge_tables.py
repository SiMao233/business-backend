"""add knowledge tables (knowledge_base / knowledge_document / knowledge_chunk)

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-09-05 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: str | None = 'd4e5f6a7b8c9'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 知识库表
    op.create_table(
        'sys_knowledge_base',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('code', sa.String(length=64), nullable=False),
        sa.Column('description', sa.String(length=512), nullable=True),
        sa.Column('icon_id', sa.Uuid(), nullable=True),
        sa.Column('organization_id', sa.Uuid(), nullable=True),
        sa.Column('embedding_model_id', sa.Uuid(), nullable=True),
        sa.Column('status', sa.SmallInteger(), nullable=False),
        sa.Column('creator_id', sa.Uuid(), nullable=True),
        sa.Column('create_time', sa.DateTime(), nullable=False),
        sa.Column('update_time', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['icon_id'], ['sys_file.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['organization_id'], ['sys_organization.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['embedding_model_id'], ['sys_model_instance.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['creator_id'], ['sys_user.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('code'),
    )
    op.create_index('ix_sys_knowledge_base_code', 'sys_knowledge_base', ['code'])
    op.create_index('ix_sys_knowledge_base_organization_id', 'sys_knowledge_base', ['organization_id'])

    # 知识库文档表
    op.create_table(
        'sys_knowledge_document',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('knowledge_base_id', sa.Uuid(), nullable=False),
        sa.Column('file_id', sa.Uuid(), nullable=True),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('chunk_count', sa.Integer(), nullable=False),
        sa.Column('error_message', sa.String(length=512), nullable=True),
        sa.Column('uploader_id', sa.Uuid(), nullable=True),
        sa.Column('create_time', sa.DateTime(), nullable=False),
        sa.Column('update_time', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['knowledge_base_id'], ['sys_knowledge_base.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['file_id'], ['sys_file.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['uploader_id'], ['sys_user.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_sys_knowledge_document_knowledge_base_id', 'sys_knowledge_document', ['knowledge_base_id'])

    # 文档切分块表（文本双写，向量落 Qdrant）
    op.create_table(
        'sys_knowledge_chunk',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('document_id', sa.Uuid(), nullable=False),
        sa.Column('knowledge_base_id', sa.Uuid(), nullable=False),
        sa.Column('seq_no', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.Column('vector_id', sa.String(length=64), nullable=True),
        sa.Column('char_count', sa.BigInteger(), nullable=True),
        sa.Column('create_time', sa.DateTime(), nullable=False),
        sa.Column('update_time', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['sys_knowledge_document.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['knowledge_base_id'], ['sys_knowledge_base.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_sys_knowledge_chunk_document_id', 'sys_knowledge_chunk', ['document_id'])
    op.create_index('ix_sys_knowledge_chunk_knowledge_base_id', 'sys_knowledge_chunk', ['knowledge_base_id'])


def downgrade() -> None:
    op.drop_index('ix_sys_knowledge_chunk_knowledge_base_id', table_name='sys_knowledge_chunk')
    op.drop_index('ix_sys_knowledge_chunk_document_id', table_name='sys_knowledge_chunk')
    op.drop_table('sys_knowledge_chunk')

    op.drop_index('ix_sys_knowledge_document_knowledge_base_id', table_name='sys_knowledge_document')
    op.drop_table('sys_knowledge_document')

    op.drop_index('ix_sys_knowledge_base_organization_id', table_name='sys_knowledge_base')
    op.drop_index('ix_sys_knowledge_base_code', table_name='sys_knowledge_base')
    op.drop_table('sys_knowledge_base')
