"""Alembic 迁移环境（异步模式）。

数据库连接串从应用配置读取，支持在线 / 离线两种执行方式。
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.core.config import get_settings
from app.core.database import Base

# 导入所有业务模块模型，供 autogenerate 识别差异
# TODO: 各模块模型定义完成后在此导入，例如：
#   from app.modules.iam import model  # noqa: F401
#   from app.modules.agent import model  # noqa: F401
from app.models import User, Role, UserRole, Permission, RolePermission


config = context.config

# 应用 alembic.ini 中的日志配置
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """从应用配置获取数据库连接串。"""
    return get_settings().database_url


def run_migrations_offline() -> None:
    """离线模式：仅生成 SQL，不连接数据库。"""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """在已建立的连接上执行迁移。"""
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """在线模式：使用异步引擎执行迁移。"""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """在线模式入口。"""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
