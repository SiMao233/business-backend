"""系统角色管理 Pydantic Schema：请求 / 响应模型。
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from app.common.schema import ApiInModel, ApiOutModel, UtcDateTime
from app.modules.system.permission.schema import PermissionOut


class RoleCreate(BaseModel):
    """创建角色的入参。

    注意：`is_builtin` 由系统（seed）控制，界面创建一律为 False，不可自建"内置"角色。
    """

    code: str = Field(min_length=1, max_length=50, description="角色编码，如 admin")
    name: str = Field(min_length=1, max_length=50, description="角色名称")
    description: str | None = Field(default=None, max_length=255, description="描述")
    status: int = Field(default=1, ge=0, le=1, description="1启用 0禁用")


class RoleUpdate(BaseModel):
    """更新角色的入参（编码一般不允许改，所以没有 code）。"""

    name: str = Field(min_length=1, max_length=50, description="角色名称")
    description: str | None = Field(default=None, max_length=255, description="描述")
    status: int = Field(default=1, ge=0, le=1, description="1启用 0禁用")


class RoleOut(ApiOutModel):
    """角色的出参（从 ORM 对象直接转换）。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    code: str
    name: str
    description: str | None = None
    status: int
    is_builtin: bool = False
    permissions: list[PermissionOut] = Field(default_factory=list)
    create_time: UtcDateTime
    update_time: UtcDateTime

    @field_serializer("id")
    def serialize_id(self, value: UUID) -> str:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex

class RolePermissionsReq(ApiInModel):
    """给角色分配权限的入参（全量覆盖式）。"""

    permission_ids: list[UUID] = Field(
        default_factory=list,
        description="权限 ID 列表（覆盖式，需提交完整列表）",
    )

class RoleQuery(BaseModel):
    """角色多条件查询（POST body）。"""

    page: int = Field(default=1, ge=1, description="页码（从 1 开始）")
    pageSize: int = Field(default=20, ge=10, le=100, description="每页条数")
    name: str | None = Field(default=None, max_length=50, description="关键字：名称/编码模糊匹配")
    code: str | None = Field(default=None, max_length=50, description="角色编码精确匹配")
    status: int | None = Field(default=None, ge=0, le=1, description="状态筛选：1启用 0禁用")
