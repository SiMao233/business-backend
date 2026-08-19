"""系统权限管理 Pydantic Schema：请求 / 响应模型。

权限采用「树状组织 + 扁平权限码」双重模型：管理端按树展示与分配，鉴权直接比对 code。
type 语义：1=目录 2=菜单 3=按钮。
"""

from uuid import UUID

from pydantic import ConfigDict, Field, field_serializer

from app.common.schema import ApiInModel, ApiOutModel, UtcDateTime


class PermissionCreate(ApiInModel):
    """创建权限点的入参。

    注意：`is_builtin` 由系统（seed）控制，界面创建一律为 False，不可自建"内置"权限。
    """

    code: str = Field(min_length=1, max_length=100, description="权限码，如 system:user:create")
    name: str = Field(min_length=1, max_length=100, description="权限名称")
    type: int = Field(default=1, ge=1, le=3, description="类型：1目录 2菜单 3按钮")
    parent_id: UUID | None = Field(default=None, description="上级权限 ID（根节点为空）")
    description: str | None = Field(default=None, max_length=255, description="描述")
    status: int = Field(default=1, ge=0, le=1, description="1启用 0禁用")


class PermissionUpdate(ApiInModel):
    """更新权限点的入参（权限码创建后不可修改）。"""

    name: str = Field(min_length=1, max_length=100, description="权限名称")
    type: int = Field(default=1, ge=1, le=3, description="类型：1目录 2菜单 3按钮")
    parent_id: UUID | None = Field(default=None, description="上级权限 ID（根节点为空）")
    description: str | None = Field(default=None, max_length=255, description="描述")
    status: int = Field(default=1, ge=0, le=1, description="1启用 0禁用")


class PermissionOut(ApiOutModel):
    """权限点的出参（从 ORM 对象直接转换）。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    code: str
    name: str
    type: int
    parent_id: UUID | None = None
    description: str | None = None
    status: int
    is_builtin: bool = False
    role_count: int = 0
    create_time: UtcDateTime
    update_time: UtcDateTime

    @field_serializer("id")
    def serialize_id(self, value: UUID) -> str:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex

    @field_serializer("parent_id")
    def serialize_parent_id(self, value: UUID | None) -> str | None:
        """上级 ID 同样输出不带连字符的 32 位字符串，空则输出 null。"""
        return value.hex if value else None


class PermissionNode(PermissionOut):
    """权限树节点：在平铺字段基础上携带子节点列表。"""

    children: list["PermissionNode"] = Field(default_factory=list)


class PermissionQuery(ApiInModel):
    """权限多条件查询（POST body）。"""

    page: int = Field(default=1, ge=1, description="页码（从 1 开始）")
    pageSize: int = Field(default=20, ge=10, le=100, description="每页条数")
    keyword: str | None = Field(default=None, max_length=100, description="关键字：名称/权限码模糊匹配")
    type: int | None = Field(default=None, ge=1, le=3, description="类型筛选：1目录 2菜单 3按钮")
    status: int | None = Field(default=None, ge=0, le=1, description="状态筛选：1启用 0禁用")


class UserPermissionOut(ApiOutModel):
    """当前用户聚合权限：权限码集合 + 菜单树（供前端渲染菜单 / 按钮鉴权）。"""

    codes: list[str] = Field(default_factory=list)
    tree: list[PermissionNode] = Field(default_factory=list)
