"""Agent Pydantic Schema：请求 / 响应模型。"""

from uuid import UUID

from pydantic import ConfigDict, Field, field_serializer

from app.common.enums import AgentStatus
from app.common.schema import ApiInModel, ApiOutModel, UtcDateTime


class AgentConfig(ApiInModel):
    """Agent 核心配置（值对象，JSON 序列化存储）。

    继承 ApiInModel：入参支持驼峰（systemPrompt / maxTokens / knowledgeIds），
    同时兼容 snake_case；出参统一序列化为驼峰，与 API 风格一致。
    """

    system_prompt: str = Field(default="", description="系统提示词")
    temperature: float = Field(default=0.7, ge=0, le=2, description="采样温度")
    max_tokens: int = Field(default=2048, ge=1, description="最大输出 token")
    tools: list[str] = Field(default_factory=list, description="启用的工具列表")
    knowledge_ids: list[UUID] = Field(default_factory=list, description="关联知识库 ID 列表")
    memory: dict = Field(default_factory=dict, description="记忆配置")

    @field_serializer("knowledge_ids")
    def serialize_knowledge_ids(self, value: list[UUID]) -> list[str]:
        """输出不带连字符的 32 位十六进制字符串（与全站 UUID 出参约定一致）。"""
        return [v.hex for v in value]


class AgentCreate(ApiInModel):
    """创建 Agent 入参（新建即草稿，状态由生命周期动作管理）。"""

    name: str = Field(min_length=1, max_length=255, description="Agent 名称")
    code: str = Field(min_length=1, max_length=64, description="业务编码")
    description: str | None = Field(default=None, max_length=512, description="描述")
    icon_id: UUID | None = Field(default=None, description="图标文件 ID")
    model_id: UUID | None = Field(default=None, description="绑定模型实例 ID")
    organization_id: UUID | None = Field(default=None, description="归属组织 ID")
    config: AgentConfig = Field(default_factory=AgentConfig, description="核心配置")


class AgentUpdate(ApiInModel):
    """更新 Agent 入参（code 不可改；配置仅草稿可改，状态可直接切换）。"""

    name: str | None = Field(default=None, min_length=1, max_length=255, description="Agent 名称")
    description: str | None = Field(default=None, max_length=512, description="描述")
    icon_id: UUID | None = Field(default=None, description="图标文件 ID")
    model_id: UUID | None = Field(default=None, description="绑定模型实例 ID")
    organization_id: UUID | None = Field(default=None, description="归属组织 ID")
    config: AgentConfig | None = Field(default=None, description="核心配置")
    status: AgentStatus | None = Field(
        default=None, description="状态切换（draft/running/paused/stopped）"
    )


class AgentOut(ApiOutModel):
    """Agent 出参。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    code: str
    description: str | None = None
    icon_id: UUID | None = None
    model_id: UUID | None = None
    organization_id: UUID | None = None
    config: AgentConfig
    status: AgentStatus
    current_version: str
    creator_id: UUID | None = None
    create_time: UtcDateTime
    update_time: UtcDateTime

    @field_serializer("id", "icon_id", "model_id", "organization_id", "creator_id")
    def serialize_id(self, value: UUID | None) -> str | None:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex if value else None


class AgentQuery(ApiInModel):
    """Agent 分页查询。"""

    page: int = Field(default=1, ge=1, description="页码（从 1 开始）")
    pageSize: int = Field(default=20, ge=10, le=100, description="每页条数")
    keyword: str | None = Field(default=None, max_length=64, description="关键字：名称/编码模糊匹配")
    status: AgentStatus | None = Field(default=None, description="状态筛选")
    organization_id: UUID | None = Field(default=None, description="按组织筛选")


class AgentPublish(ApiInModel):
    """发布 Agent 入参。"""

    version: str = Field(min_length=1, max_length=64, description="版本号（用户自定义，如 v1.0.0）")
    changelog: str | None = Field(default=None, max_length=512, description="发布说明")


class AgentVersionSwitch(ApiInModel):
    """切换 Agent 版本入参（回滚到历史版本）。"""

    version: str = Field(min_length=1, max_length=64, description="目标版本号")


class AgentVersionOut(ApiOutModel):
    """Agent 版本出参。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    agent_id: UUID
    version: str
    config: AgentConfig
    changelog: str | None = None
    status: str
    create_time: UtcDateTime
    update_time: UtcDateTime

    @field_serializer("id", "agent_id")
    def serialize_id(self, value: UUID) -> str:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex


class AgentRunRequest(ApiInModel):
    """触发运行入参。"""

    input: str = Field(min_length=1, max_length=10000, description="用户输入")


class AgentRunOut(ApiOutModel):
    """触发运行回执（本版不建 run 记录表，同步返回对话结果）。"""

    run_id: str = Field(default="", description="运行 ID（预留，异步时返回）")
    status: str = Field(default="success", description="运行状态")
    reply: str = Field(default="", description="模型回复内容")
    model: str = Field(default="", description="实际使用的模型标识")
