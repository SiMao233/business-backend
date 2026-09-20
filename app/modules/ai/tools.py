"""AI 工具注册表：把 Agent 配置的 tools 映射为 LangChain 工具。

设计约束：
- 工具实现收敛在本模块，业务模块不直接依赖 langchain；
- 工具通过工厂函数构造（闭包注入 config 上下文）；
- 工具名（函数名）必须与 AgentConfig.tools 里的字符串一致；
- 工具内部按需自开会话（AsyncSessionLocal）：流式对话场景下请求级会话已归还，
  工具不能依赖外部传入的会话，因此统一在调用时开短会话。

来源与引用编号（`cited`）：知识库检索工具用 `response_format="content_and_artifact"`，
把结构化来源放进 `ToolMessage.artifact`（**不发给模型**），文本仍按 `format_context` 呈现给模型。
两块结果共用调用方传入的 `cited` 计数器，保证 `[n]` 在**单次请求内全局唯一**：
预检索块占 1..k，模型后续调工具检索到的资料从 k+1 继续编号，正文角标不会二义。

空命中不返回空串，而是 `grounding.NO_EVIDENCE_TOOL_TEXT`：显式告诉模型「本轮没有资料」，
否则模型会把空返回理解为「不受约束」，转用自己的预训练知识作答（正是要堵的幻觉来源）。
"""

from langchain_core.tools import tool

from app.modules.ai.grounding import NO_EVIDENCE_TOOL_TEXT


def build_tools(config: dict, cited: dict[str, int] | None = None) -> list:
    """根据 config.tools 构建 LangChain 工具列表（工具名与配置字符串一致）。

    `cited`：请求级的「已呈现给模型的资料条数」计数器（初值 `{"n": 0}`），
    由调用方（chat service）创建并同时用于预检索，使两处 `[n]` 编号首尾相接。
    不传则自建（单次调用场景，如未来的独立工具调用）。
    """
    if cited is None:
        cited = {"n": 0}
    enabled = set(config.get("tools") or [])
    tools = []
    if "knowledge_retrieval" in enabled:
        tools.append(_knowledge_retrieval_tool(config, cited))
    if "web_search" in enabled:
        tools.append(_web_search_tool())
    return tools


def _knowledge_retrieval_tool(config: dict, cited: dict[str, int]):
    """知识库检索：调用 retrieval 的检索 + 拼装（闭包注入 config 与 cited，每次调用自开短会话）。

    返回 `(参考资料文本, 来源列表)`：文本进模型上下文，来源进 `ToolMessage.artifact`，
    供 `steps[].sources` 落库与前端渲染来源卡片。
    """

    @tool(response_format="content_and_artifact")
    async def knowledge_retrieval(query: str) -> tuple[str, list[dict]]:
        """从 Agent 绑定的知识库中检索与问题相关的资料片段，返回参考资料文本；无结果返回空串。"""
        # 延迟导入：避免模块导入期的循环依赖
        from app.core.database import AsyncSessionLocal
        from app.modules.ai.retrieval import build_sources, format_context, retrieve_hits

        async with AsyncSessionLocal() as db:
            hits = await retrieve_hits(db, config, query)
            # 空命中：显式告知「没有资料」，并保持编号计数不变（不占用 [n]）
            if not hits:
                return NO_EVIDENCE_TOOL_TEXT, []
            # 读改写必须紧邻（中间不得有 await）：asyncio 协作式调度下即为原子操作，
            # 故模型并发发起多次工具调用也不会拿到重复编号。
            start = cited["n"]
            cited["n"] += len(hits)
            return format_context(hits, start), build_sources(hits, start)

    return knowledge_retrieval


def _web_search_tool():
    """联网搜索工具（TODO：接入搜索服务后实现，如 Tavily / 阿里云百炼 / 博查）。"""

    @tool
    async def web_search(query: str) -> str:
        """联网搜索最新信息，返回搜索结果摘要（综合回答 + 标题/链接/摘要）。"""
        # TODO: 接入搜索服务（Tavily / 阿里云百炼 / 博查），当前为占位实现
        return "联网搜索功能尚未实现"

    return web_search
