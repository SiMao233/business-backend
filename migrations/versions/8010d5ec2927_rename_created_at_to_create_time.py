"""rename created_at to create_time

Revision ID: 8010d5ec2927
Revises: f47c6811a591
Create Date: 2026-08-13 00:32:05.967160

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8010d5ec2927'
down_revision: Union[str, None] = 'f47c6811a591'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# 所有继承 TimestampMixin 的表，时间列统一由 created_at/updated_at 重命名为 create_time/update_time
_TABLES = [
    "sys_user",
    "sys_role",
    "sys_permission",
    "sys_user_role",
    "sys_role_permission",
]


def upgrade() -> None:
    for table in _TABLES:
        op.alter_column(table, "created_at", new_column_name="create_time", existing_type=sa.DateTime())
        op.alter_column(table, "updated_at", new_column_name="update_time", existing_type=sa.DateTime())


def downgrade() -> None:
    for table in _TABLES:
        op.alter_column(table, "create_time", new_column_name="created_at", existing_type=sa.DateTime())
        op.alter_column(table, "update_time", new_column_name="updated_at", existing_type=sa.DateTime())
