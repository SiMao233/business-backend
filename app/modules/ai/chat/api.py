"""AI 对话 API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。

本层禁止直接操作数据库或编写复杂业务逻辑。路由风格：动作前置（如 /ai/chat）。

`POST /ai/chat` 为 **SSE 流式**接口（text/event-stream）：
- 事件协议：`meta` →（`tool`）→ `delta`… → `done`；流中途错误以 `error` 事件返回；
- 预检（Agent/会话校验、落库 user 消息）放在依赖 `prepare_chat_stream` 中执行——
  SSE 端点必须是生成器函数，其函数体在「响应已开始」之后才运行，那时抛错已无法转成
  JSON 错误响应；依赖在响应创建前解析，失败仍由全局异常处理器返回统一 JSON 错误体。
"""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.authentication import CurrentUserDep, get_current_user
from app.middleware.permission import require_permissions
from app.modules.ai.chat.schema import AiChatRequest
from app.modules.ai.chat.service import AiChatService, ChatStreamContext, stream_chat
from app.modules.ai.codes import PermissionCode

router = APIRouter(prefix="/chat", tags=["AI能力"], dependencies=[Depends(get_current_user)])

DbDep = Annotated[AsyncSession, Depends(get_db)]


async def prepare_chat_stream(
    req: AiChatRequest,
    current: CurrentUserDep,
    db: DbDep,
) -> ChatStreamContext:
    """流式对话预检依赖：校验并准备上下文，随后立即归还数据库连接。

    为何要归还连接：SSE 连接可能持续数十秒以上，若整段流都持有请求级会话，
    会长时间占用连接池（pool_size=10 + max_overflow=20），高并发下拖垮其他接口。
    """
    ctx = await AiChatService(db).prepare_chat(
        req.agent_id, req.input, current.user_id, req.session_id
    )
    await db.close()
    return ctx


# 触发 Agent 对话（SSE 流式）：事件 meta → (tool) → delta… → done
@router.post(
    "",
    response_class=EventSourceResponse,
    summary="Agent 对话（SSE 流式）",
    dependencies=[Depends(require_permissions(PermissionCode.AI_CHAT))],
)
async def ai_chat(
    ctx: Annotated[ChatStreamContext, Depends(prepare_chat_stream)],
) -> AsyncIterator[ServerSentEvent]:
    async for event in stream_chat(ctx):
        # 显式按别名（camelCase）序列化：FastAPI 的 SSE 编码默认走 model_dump_json()，
        # 而本项目的驼峰别名只在 by_alias=True 时生效，故此处先转 dict 再交给 SSE 编码。
        yield ServerSentEvent(event=event.type, data=event.data.model_dump(by_alias=True))
