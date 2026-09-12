"""保留第三方 OpenAI 兼容网关返回的推理内容（reasoning_content）。

官方 langchain-openai 的 ChatOpenAI 明确不解析/不保留第三方网关的非标准字段
（reasoning_content / reasoning_details 等，见其模块 docstring 警告）。
但 openai SDK 的响应模型是 extra="allow"，langchain 在流式/非流式转换前都会
`chunk.model_dump()`，原始 dict 里其实**带着**这些字段，只是转换函数没读。

这里子类化 ChatOpenAI，在转换后把推理内容补进 `additional_kwargs`，
上层（chat/service.py）统一从 `additional_kwargs` 提取，无需关心网关差异。
网关不返回推理时静默跳过，不影响原有行为（兼容式设计）。
"""

from typing import Any

from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI

# 常见 OpenAI 兼容网关的推理字段名（按优先级取第一个非空值）
REASONING_KEYS = ("reasoning_content", "reasoning", "reasoning_details", "thinking")


def _extract_reasoning(raw: dict[str, Any]) -> str:
    """从原始响应 dict 中提取推理文本（兼容多种网关字段名）。"""
    for key in REASONING_KEYS:
        value = raw.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


class ReasoningChatOpenAI(ChatOpenAI):
    """在 additional_kwargs 中保留推理内容的 ChatOpenAI（流式 + 非流式）。"""

    # 流式路径：逐 chunk 把 delta.reasoning_content 补进 AIMessageChunk
    def _convert_chunk_to_generation_chunk(
        self, chunk: dict, default_chunk_class: type, base_generation_info: dict | None
    ) -> Any:
        gen = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if gen is None or not isinstance(gen.message, AIMessageChunk):
            return gen
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            if reasoning := _extract_reasoning(delta):
                gen.message.additional_kwargs["reasoning_content"] = reasoning
        return gen

    # 非流式路径：从 choices[0].message 补进最终 AIMessage
    def _create_chat_result(self, response: Any, generation_info: dict | None = None) -> Any:
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        choices = raw.get("choices") or []
        if choices and result.generations:
            message = choices[0].get("message") or {}
            if reasoning := _extract_reasoning(message):
                result.generations[0].message.additional_kwargs["reasoning_content"] = reasoning
        return result
