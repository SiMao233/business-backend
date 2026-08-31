"""AI 能力 API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。

本层禁止直接操作数据库或编写复杂业务逻辑。
路由风格：动作前置（如 /ai/chat）。
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.response import ApiResponse, success
from app.core.database import get_db
from app.middleware.authentication import CurrentUserDep, get_current_user
from app.middleware.permission import require_permissions
from app.modules.ai.codes import PermissionCode
from app.modules.ai.schema import AiChatOut, AiChatRequest
from app.modules.ai.service import AiService

router = APIRouter(prefix="/ai", tags=["AI能力"], dependencies=[Depends(get_current_user)])

DbDep = Annotated[AsyncSession, Depends(get_db)]


# 依赖注入工厂：创建 AiService 实例（绑定数据库会话）
def get_service(db: DbDep) -> AiService:
    return AiService(db)


# 触发 Agent 对话：同步返回模型回复
@router.post(
    "/chat",
    response_model=ApiResponse[AiChatOut],
    summary="Agent 对话",
    dependencies=[Depends(require_permissions(PermissionCode.AI_CHAT))],
)
async def ai_chat(
    req: AiChatRequest,
    current: CurrentUserDep,
    service: Annotated[AiService, Depends(get_service)],
) -> ApiResponse[AiChatOut]:
    return success(data=await service.chat(req.agent_id, req.input, current.user_id))
