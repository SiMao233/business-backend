"""统一 API 响应模型。

业务约定：`code == 0` 表示成功，非 0 表示业务错误。
所有接口（含异常）统一返回该结构，便于前端统一处理。
"""

from typing import Any

from pydantic import BaseModel

SUCCESS_CODE = 200
ERROR_CODE = 500


class ApiResponse[T](BaseModel):
    """统一响应包装。"""

    code: int = SUCCESS_CODE
    message: str = "success"
    data: T | None = None


def success(data: Any = None, message: str = "success") -> ApiResponse[Any]:
    """构造成功响应。"""
    return ApiResponse(code=SUCCESS_CODE, message=message, data=data)


def error(code: int = ERROR_CODE, message: str = "error", data: Any = None) -> ApiResponse[Any]:
    """构造失败响应。"""
    return ApiResponse(code=code, message=message, data=data)
