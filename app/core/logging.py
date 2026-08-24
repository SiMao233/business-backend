"""日志配置：基于 loguru 的统一日志输出。

- 所有日志输出到 stdout（容器/进程标准输出，便于 Docker 与日志平台收集）
- 拦截标准库 logging（uvicorn / sqlalchemy 等）统一汇入 loguru
- 生产环境输出 JSON 结构化日志，便于 ELK / Loki 等平台解析
"""

import logging
import sys

from loguru import logger

from app.core.config import get_settings

settings = get_settings()

# 移除 loguru 默认 sink，避免重复输出
logger.remove()

# 开发环境可读格式：时间 | 级别 | 模块:函数:行号 - 消息
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


class InterceptHandler(logging.Handler):
    """将标准库 logging 记录桥接到 loguru，统一输出格式。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging() -> None:
    """配置全局日志：stdout sink + 标准库桥接。应用启动时调用一次。"""
    level = settings.log_level.upper()
    if settings.app_env == "production":
        # 生产：JSON 结构化输出，便于日志平台解析
        logger.add(sys.stdout, level=level, serialize=True)
    else:
        logger.add(sys.stdout, level=level, format=LOG_FORMAT, colorize=True)

    # 标准库 logging -> loguru 桥接
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "sqlalchemy.engine"):
        lg = logging.getLogger(name)
        lg.handlers = [InterceptHandler()]
        lg.propagate = False
    # 访问日志由自定义中间件输出（含耗时），屏蔽 uvicorn 自带的
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)