"""组织管理 Pydantic Schema：请求 / 响应模型。"""

from uuid import UUID

from pydantic import ConfigDict, Field, field_serializer

from app.common.schema import ApiInModel, ApiOutModel, UtcDateTime


class OrganizationCreate(ApiInModel):
    """创建组织入参。"""

    name: str = Field(min_length=1, max_length=64, description="组织名称")
    code: str = Field(min_length=1, max_length=64, description="业务编码")
    description: str | None = Field(default=None, max_length=255, description="描述")
    parent_id: UUID | None = Field(default=None, description="上级组织 ID")
    owner_id: UUID | None = Field(default=None, description="负责人 ID")
    status: int = Field(default=1, ge=0, le=1, description="1启用 0禁用")


class OrganizationUpdate(ApiInModel):
    """更新组织入参（code 不可改）。"""

    name: str | None = Field(default=None, min_length=1, max_length=64, description="组织名称")
    description: str | None = Field(default=None, max_length=255, description="描述")
    parent_id: UUID | None = Field(default=None, description="上级组织 ID")
    owner_id: UUID | None = Field(default=None, description="负责人 ID")
    status: int | None = Field(default=None, ge=0, le=1, description="1启用 0禁用")


class OrganizationOut(ApiOutModel):
    """组织出参。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    parent_id: UUID | None = None
    name: str
    code: str
    description: str | None = None
    owner_id: UUID | None = None
    status: int
    create_time: UtcDateTime
    update_time: UtcDateTime

    @field_serializer("id", "parent_id", "owner_id")
    def serialize_id(self, value: UUID | None) -> str | None:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex if value else None


class OrganizationNode(ApiOutModel):
    """组织树节点（children 递归）。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    parent_id: UUID | None = None
    name: str
    code: str
    description: str | None = None
    owner_id: UUID | None = None
    status: int
    children: list["OrganizationNode"] = Field(default_factory=list, description="子组织")

    @field_serializer("id", "parent_id", "owner_id")
    def serialize_id(self, value: UUID | None) -> str | None:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex if value else None


class OrganizationQuery(ApiInModel):
    """组织分页查询。"""

    page: int = Field(default=1, ge=1, description="页码（从 1 开始）")
    pageSize: int = Field(default=20, ge=10, le=100, description="每页条数")
    keyword: str | None = Field(default=None, max_length=64, description="关键字：名称/编码模糊匹配")
    status: int | None = Field(default=None, ge=0, le=1, description="状态筛选")


# 递归模型需在类定义后重建引用
OrganizationNode.model_rebuild()
