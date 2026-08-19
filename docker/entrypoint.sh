#!/bin/sh
# 应用容器入口：执行数据库迁移、初始化基础数据后启动 Uvicorn
set -e

echo "[entrypoint] Running database migrations..."
alembic upgrade head

echo "[entrypoint] Seeding initial data (idempotent)..."
python -m app.core.seed

echo "[entrypoint] Starting application..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
