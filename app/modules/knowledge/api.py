"""Knowledge API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。

本层禁止直接操作数据库或编写复杂业务逻辑。
TODO(knowledge): 知识库、文档管理等接口。
"""

from fastapi import APIRouter, Depends

from app.middleware.authentication import get_current_user

router = APIRouter(prefix="/knowledge", tags=["知识库"], dependencies=[Depends(get_current_user)])
