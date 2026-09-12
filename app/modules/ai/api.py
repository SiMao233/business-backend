"""AI 能力 API 聚合入口。

作为 ai 域的统一入口：
- 挂载对话（chat）与会话（conversation）子模块路由；
- 子模块 api 通过相对 prefix（/chat、/conversations）被本文件 include_router 聚合，
  最终 URL 形如 /api/v1/ai/chat、/api/v1/ai/conversations/list。
"""

from fastapi import APIRouter

from app.modules.ai.chat.api import router as chat_router
from app.modules.ai.conversation.api import router as conversation_router

# AI 能力总路由（URL 前缀 /ai）
router = APIRouter(prefix="/ai")

# 聚合子模块路由：/ai/chat、/ai/conversations/*
router.include_router(chat_router)
router.include_router(conversation_router)
