"""AI 工具注册表：把 Agent 配置的 tools 映射为 LangChain 工具。

设计约束：
- 工具实现收敛在本模块，业务模块不直接依赖 langchain；
- 工具通过工厂函数构造（闭包注入 config 上下文）；
- 工具名（函数名）必须与 AgentConfig.tools 里的字符串一致；
- 工具内部按需自开会话（AsyncSessionLocal）：流式对话场景下请求级会话已归还，
  工具不能依赖外部传入的会话，因此统一在调用时开短会话。
"""

from langchain_core.tools import tool


def build_tools(config: dict) -> list:
    """根据 config.tools 构建 LangChain 工具列表（工具名与配置字符串一致）。"""
    enabled = set(config.get("tools") or [])
    tools = []
    if "knowledge_retrieval" in enabled:
        tools.append(_knowledge_retrieval_tool(config))
    if "web_search" in enabled:
        tools.append(_web_search_tool())
    return tools


def _knowledge_retrieval_tool(config: dict):
    """知识库检索：复用 retrieval.retrieve_context（闭包注入 config，每次调用自开短会话）。"""

    @tool
    async def knowledge_retrieval(query: str) -> str:
        """从 Agent 绑定的知识库中检索与问题相关的资料片段，返回参考资料文本；无结果返回空串。"""
        # 延迟导入：避免模块导入期的循环依赖
        from app.core.database import AsyncSessionLocal
        from app.modules.ai.retrieval import retrieve_context

        async with AsyncSessionLocal() as db:
            return await retrieve_context(db, config, query)

    return knowledge_retrieval


def _web_search_tool():
    """联网搜索工具（TODO：接入搜索服务后实现，如 Tavily / 阿里云百炼 / 博查）。"""

    @tool
    async def web_search(query: str) -> str:
        """联网搜索最新信息，返回搜索结果摘要（综合回答 + 标题/链接/摘要）。"""
        # TODO: 接入搜索服务（Tavily / 阿里云百炼 / 博查），当前为占位实现
        return "联网搜索功能尚未实现"

    return web_search
