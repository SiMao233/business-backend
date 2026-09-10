"""AI 能力 API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。

本层禁止直接操作数据库或编写复杂业务逻辑。
路由风格：动作前置（如 /ai/chat）；静态路径（/conversations/list /create）注册在 /{id} 之前。
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageResult
from app.common.response import ApiResponse, success
from app.core.database import get_db
from app.middleware.authentication import CurrentUserDep, get_current_user
from app.middleware.permission import require_permissions
from app.modules.ai.codes import PermissionCode
from app.modules.ai.schema import (
    AiChatOut,
    AiChatRequest,
    ConversationCreate,
    ConversationOut,
    ConversationQuery,
    MessageOut,
)
from app.modules.ai.service import AiService

router = APIRouter(prefix="/ai", tags=["AI能力"], dependencies=[Depends(get_current_user)])

DbDep = Annotated[AsyncSession, Depends(get_db)]


# 依赖注入工厂：创建 AiService 实例（绑定数据库会话）
def get_service(db: DbDep) -> AiService:
    return AiService(db)


# 触发 Agent 对话：同步返回模型回复（RAG 检索 + 多轮记忆）
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
    return success(data=await service.chat(req.agent_id, req.input, current.user_id, req.session_id))


# ---- 会话管理（静态路径注册在 /{conversation_id} 之前）----
@router.post(
    "/conversations/list",
    response_model=ApiResponse[PageResult[ConversationOut]],
    summary="会话列表",
    dependencies=[Depends(require_permissions(PermissionCode.CONVERSATION_LIST))],
)
# 会话列表：分页查询当前用户的会话
async def list_conversations(
    query: ConversationQuery,
    current: CurrentUserDep,
    service: Annotated[AiService, Depends(get_service)],
) -> ApiResponse[PageResult[ConversationOut]]:
    return success(data=await service.list_conversations(query, current.user_id))


@router.post(
    "/conversations/create",
    response_model=ApiResponse[ConversationOut],
    summary="创建会话",
    dependencies=[Depends(require_permissions(PermissionCode.CONVERSATION_CREATE))],
)
# 创建会话：手动新建（也可通过 /ai/chat 自动创建）
async def create_conversation(
    req: ConversationCreate,
    current: CurrentUserDep,
    service: Annotated[AiService, Depends(get_service)],
) -> ApiResponse[ConversationOut]:
    return success(data=await service.create_conversation(req, current.user_id), message="创建成功")


@router.get(
    "/conversations/{conversation_id}",
    response_model=ApiResponse[ConversationOut],
    summary="会话详情",
    dependencies=[Depends(require_permissions(PermissionCode.CONVERSATION_LIST))],
)
# 会话详情：按 ID 查询单个会话
async def get_conversation(
    conversation_id: UUID,
    current: CurrentUserDep,
    service: Annotated[AiService, Depends(get_service)],
) -> ApiResponse[ConversationOut]:
    return success(data=await service.get_conversation(conversation_id, current.user_id))


@router.delete(
    "/conversations/{conversation_id}",
    response_model=ApiResponse[None],
    summary="删除会话",
    dependencies=[Depends(require_permissions(PermissionCode.CONVERSATION_DELETE))],
)
# 删除会话：按 ID 删除（消息级联删除）
async def delete_conversation(
    conversation_id: UUID,
    current: CurrentUserDep,
    service: Annotated[AiService, Depends(get_service)],
) -> ApiResponse[None]:
    await service.delete_conversation(conversation_id, current.user_id)
    return success(message="删除成功")


@router.get(
    "/conversations/messages/{conversation_id}",
    response_model=ApiResponse[list[MessageOut]],
    summary="会话消息列表",
    dependencies=[Depends(require_permissions(PermissionCode.CONVERSATION_LIST))],
)
# 会话消息列表：按 ID 查询该会话的全部消息（时间升序）
async def list_messages(
    conversation_id: UUID,
    current: CurrentUserDep,
    service: Annotated[AiService, Depends(get_service)],
) -> ApiResponse[list[MessageOut]]:
    return success(data=await service.list_messages(conversation_id, current.user_id))
