# Business Backend 应用镜像（多阶段构建，基于 uv 管理依赖）
# 构建方式：docker build -f docker/Dockerfile -t business-backend .

# ---------- 基础阶段 ----------
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# 安装 uv（官方镜像内置 /uv 与 /uvx）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# ---------- 依赖阶段（利用构建缓存） ----------
FROM base AS dependencies

# 先复制依赖清单，仅在清单变更时重新解析依赖
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ---------- 运行阶段 ----------
FROM base AS runtime

COPY --from=dependencies /app/.venv /app/.venv

# 复制项目代码与迁移脚本
COPY app ./app
COPY migrations ./migrations
COPY alembic.ini ./

# 入口脚本：先执行迁移，再启动服务
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
