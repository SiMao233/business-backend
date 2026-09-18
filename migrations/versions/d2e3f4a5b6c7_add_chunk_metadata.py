"""add chunk metadata (section_path / page_no / chunk_type)

Revision ID: d2e3f4a5b6c7
Revises: c1d2e3f4a5b6
Create Date: 2026-09-15 00:00:00.000000

结构感知切分产出的块元数据：
- section_path：块所属章节路径（如「员工手册 > 二、考勤」），无标题结构时为 NULL
- page_no：来源页码（PDF 等分页文档；跨页块取起始页）
- chunk_type：块类型（text / table / mixed）

存量数据兼容：chunk_type 用 server_default='text' 回填，另两列可空 —— 旧 chunk
不受影响，重新索引时会按新流程重算这三列。
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd2e3f4a5b6c7'
down_revision: str | None = 'c1d2e3f4a5b6'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'sys_knowledge_chunk',
        sa.Column('section_path', sa.String(length=512), nullable=True),
    )
    op.add_column(
        'sys_knowledge_chunk',
        sa.Column('page_no', sa.Integer(), nullable=True),
    )
    op.add_column(
        'sys_knowledge_chunk',
        sa.Column('chunk_type', sa.String(length=16), nullable=False, server_default='text'),
    )


def downgrade() -> None:
    op.drop_column('sys_knowledge_chunk', 'chunk_type')
    op.drop_column('sys_knowledge_chunk', 'page_no')
    op.drop_column('sys_knowledge_chunk', 'section_path')
