"""Organization API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。

本层禁止直接操作数据库或编写复杂业务逻辑。
TODO(org): 组织、团队、成员等接口。
"""

from fastapi import APIRouter, Depends

from app.middleware.authentication import get_current_user

router = APIRouter(prefix="/organization", tags=["组织管理"], dependencies=[Depends(get_current_user)])
