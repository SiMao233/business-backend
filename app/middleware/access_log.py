"""访问日志中间件：记录每个 HTTP 请求的方法、路径、状态码、耗时与客户端 IP。"""

import time

from loguru import logger


class AccessLogMiddleware:
    """纯 ASGI 中间件，避免 BaseHTTPMiddleware 对流式响应/后台任务的干扰。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status = {"code": 0}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        await self.app(scope, receive, send_wrapper)

        duration_ms = (time.perf_counter() - start) * 1000
        client_ip = scope.get("client", ("-",))[0]
        logger.info(
            "{} {} -> {} ({}ms) ip={}",
            scope["method"],
            scope["path"],
            status["code"],
            f"{duration_ms:.1f}",
            client_ip,
        )