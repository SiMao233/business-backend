"""模型管理 Pydantic Schema：请求 / 响应模型。"""

from decimal import Decimal
from uuid import UUID

from pydantic import ConfigDict, Field, field_serializer

from app.common.enums import ModelType
from app.common.schema import ApiInModel, ApiOutModel, UtcDateTime


# ---- 模型供应商 ----
class ModelProviderCreate(ApiInModel):
    """创建模型供应商入参。"""

    name: str = Field(min_length=1, max_length=64, description="供应商名称")
    code: str = Field(min_length=1, max_length=64, description="业务编码（如 openai）")
    base_url: str | None = Field(default=None, max_length=255, description="OpenAI 兼容 base_url")
    api_key: str | None = Field(default=None, max_length=512, description="API Key")
    status: int = Field(default=1, ge=0, le=1, description="1启用 0禁用")


class ModelProviderUpdate(ApiInModel):
    """更新模型供应商入参（code 不可改）。"""

    name: str | None = Field(default=None, min_length=1, max_length=64, description="供应商名称")
    base_url: str | None = Field(default=None, max_length=255, description="OpenAI 兼容 base_url")
    api_key: str | None = Field(default=None, max_length=512, description="API Key（传值则覆盖）")
    status: int | None = Field(default=None, ge=0, le=1, description="1启用 0禁用")


class ModelProviderOut(ApiOutModel):
    """模型供应商出参（api_key 不回传，仅返回 has_api_key）。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    code: str
    base_url: str | None = None
    has_api_key: bool = Field(default=False, description="是否已配置 API Key")
    is_builtin: bool = Field(default=False, description="是否内置供应商")
    status: int
    create_time: UtcDateTime
    update_time: UtcDateTime

    @field_serializer("id")
    def serialize_id(self, value: UUID) -> str:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex


class ModelProviderQuery(ApiInModel):
    """模型供应商分页查询。"""

    page: int = Field(default=1, ge=1, description="页码（从 1 开始）")
    pageSize: int = Field(default=20, ge=10, le=100, description="每页条数")
    keyword: str | None = Field(default=None, max_length=64, description="关键字：名称/编码模糊匹配")
    status: int | None = Field(default=None, ge=0, le=1, description="状态筛选")


# ---- 模型实例 ----
class ModelInstanceCreate(ApiInModel):
    """创建模型实例入参。"""

    provider_id: UUID = Field(description="所属供应商 ID")
    name: str = Field(min_length=1, max_length=64, description="展示名")
    code: str = Field(min_length=1, max_length=64, description="调用时模型标识")
    model_type: ModelType = Field(default=ModelType.CHAT, description="模型类型")
    max_tokens: int | None = Field(default=None, ge=1, description="上下文上限（token）")
    input_price: Decimal | None = Field(
        default=None, ge=0, description="输入单价（元 / 千 token，用于用量成本核算）"
    )
    output_price: Decimal | None = Field(
        default=None, ge=0, description="输出单价（元 / 千 token，用于用量成本核算）"
    )
    status: int = Field(default=1, ge=0, le=1, description="1启用 0禁用")


class ModelInstanceUpdate(ApiInModel):
    """更新模型实例入参。"""

    name: str | None = Field(default=None, min_length=1, max_length=64, description="展示名")
    code: str | None = Field(default=None, min_length=1, max_length=64, description="调用时模型标识")
    model_type: ModelType | None = Field(default=None, description="模型类型")
    max_tokens: int | None = Field(default=None, ge=1, description="上下文上限（token）")
    input_price: Decimal | None = Field(
        default=None, ge=0, description="输入单价（元 / 千 token）"
    )
    output_price: Decimal | None = Field(
        default=None, ge=0, description="输出单价（元 / 千 token）"
    )
    status: int | None = Field(default=None, ge=0, le=1, description="1启用 0禁用")


class ModelInstanceOut(ApiOutModel):
    """模型实例出参。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    provider_id: UUID
    provider_name: str | None = Field(default=None, description="供应商名称（Service 填充）")
    name: str
    code: str
    model_type: str
    max_tokens: int | None = None
    input_price: Decimal | None = Field(default=None, description="输入单价（元 / 千 token）")
    output_price: Decimal | None = Field(default=None, description="输出单价（元 / 千 token）")
    status: int
    create_time: UtcDateTime
    update_time: UtcDateTime

    @field_serializer("id", "provider_id")
    def serialize_id(self, value: UUID) -> str:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex

    @field_serializer("input_price", "output_price")
    def serialize_price(self, value: Decimal | None) -> str | None:
        """金额输出为字符串，避免 JSON 浮点精度问题。"""
        return None if value is None else format(value, "f")


class ModelInstanceQuery(ApiInModel):
    """模型实例分页查询。"""

    page: int = Field(default=1, ge=1, description="页码（从 1 开始）")
    pageSize: int = Field(default=20, ge=10, le=100, description="每页条数")
    keyword: str | None = Field(default=None, max_length=64, description="关键字：名称/编码模糊匹配")
    provider_id: UUID | None = Field(default=None, description="按供应商筛选")
    model_type: ModelType | None = Field(default=None, description="按类型筛选")
    status: int | None = Field(default=None, ge=0, le=1, description="状态筛选")


class ModelInstanceOption(ApiOutModel):
    """模型实例下拉选项（Agent 创建时选择用）。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    code: str
    model_type: str
    provider_name: str | None = None

    @field_serializer("id")
    def serialize_id(self, value: UUID) -> str:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex
