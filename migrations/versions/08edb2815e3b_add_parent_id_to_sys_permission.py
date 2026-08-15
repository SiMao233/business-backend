"""add parent_id to sys_permission

Revision ID: 08edb2815e3b
Revises: 8010d5ec2927
Create Date: 2026-08-15 00:21:15.697622

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '08edb2815e3b'
down_revision: str | None = '8010d5ec2927'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 权限树：sys_permission 增加自引用父级外键 parent_id
    op.add_column('sys_permission', sa.Column('parent_id', sa.Uuid(), nullable=True))
    op.create_index(op.f('ix_sys_permission_parent_id'), 'sys_permission', ['parent_id'], unique=False)
    op.create_foreign_key(
        'fk_sys_permission_parent_id',
        'sys_permission',
        'sys_permission',
        ['parent_id'],
        ['id'],
        ondelete='CASCADE',
    )


def downgrade() -> None:
    op.drop_constraint('fk_sys_permission_parent_id', 'sys_permission', type_='foreignkey')
    op.drop_index(op.f('ix_sys_permission_parent_id'), table_name='sys_permission')
    op.drop_column('sys_permission', 'parent_id')
