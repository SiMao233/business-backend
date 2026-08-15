"""Agent API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。

本层禁止直接操作数据库或编写复杂业务逻辑。
TODO(agent): Agent 定义、配置、启停等接口。
"""

from fastapi import APIRouter, Depends

from app.middleware.authentication import get_current_user

router = APIRouter(prefix="/agent", tags=["Agent"], dependencies=[Depends(get_current_user)])
