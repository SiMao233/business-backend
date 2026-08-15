#!/bin/sh
# 应用容器入口：执行数据库迁移后启动 Uvicorn
set -e

echo "[entrypoint] Running database migrations..."
alembic upgrade head

echo "[entrypoint] Starting application..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
