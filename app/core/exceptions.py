"""全局异常体系与统一异常处理。

业务异常约定：
- `code`：业务错误码（**成功为 200**，见 `app/common/response.py` 的 `SUCCESS_CODE`）
- `http_status`：HTTP 状态码
"""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.common.response import error


class AppError(Exception):
    """业务异常基类。"""

    def __init__(
        self,
        message: str = "业务处理失败",
        code: int = 1000,
        http_status: int = 400,
    ) -> None:
        self.message = message
        self.code = code
        self.http_status = http_status
        super().__init__(message)


class BizError(AppError):
    """通用业务错误。"""

    def __init__(self, message: str = "业务处理失败", code: int = 500) -> None:
        super().__init__(message=message, code=code, http_status=500)


class UnauthorizedError(AppError):
    """未认证 / 登录已过期。"""

    def __init__(self, message: str = "未认证或登录已过期") -> None:
        super().__init__(message=message, code=1401, http_status=401)


class PermissionDeniedError(AppError):
    """无操作权限。"""

    def __init__(self, message: str = "没有操作权限") -> None:
        super().__init__(message=message, code=1403, http_status=403)


class NotFoundError(AppError):
    """资源不存在。"""

    def __init__(self, message: str = "资源不存在") -> None:
        super().__init__(message=message, code=1404, http_status=404)


class ValidateError(AppError):
    """参数校验失败。"""

    def __init__(self, message: str = "参数校验失败", code: int = 1422) -> None:
        super().__init__(message=message, code=code, http_status=422)


def register_exception_handlers(app: FastAPI) -> None:
    """注册全局异常处理器，统一输出 `ApiResponse` 结构。"""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content=error(code=exc.code, message=exc.message).model_dump(),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error(code=exc.status_code, message=str(exc.detail)).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        first: dict[str, Any] = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(part) for part in first.get("loc", []))
        detail = str(first.get("msg", ""))
        message = f"参数校验失败: {loc} {detail}".strip()
        return JSONResponse(
            status_code=422,
            content=error(code=1422, message=message).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # 兜底：未预期的运行时异常统一返回 ApiResponse 结构，避免裸奔 traceback
        logger.exception("Unhandled exception: {} {}", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=error(code=500, message="服务器内部错误").model_dump(),
        )
