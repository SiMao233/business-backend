"""系统用户管理 Pydantic Schema：请求 / 响应模型。"""

from typing import Optional
from uuid import UUID

from pydantic import ConfigDict, Field, field_serializer

from app.common.schema import ApiInModel, ApiOutModel, UtcDateTime
from app.modules.system.role.schema import RoleOut


class UserCreate(ApiInModel):
    """创建用户的入参。"""

    username: str = Field(min_length=1, max_length=50, description="登录账号")
    password: str = Field(min_length=6, max_length=50, description="初始密码")
    nickname: Optional[str] = Field(default=None, max_length=50, description="昵称")
    email: Optional[str] = Field(default=None, max_length=100, description="邮箱")
    phone: Optional[str] = Field(default=None, max_length=20, description="手机号")
    avatar: Optional[str] = Field(default=None, max_length=500, description="头像")
    status: int = Field(default=1, ge=0, le=1, description="1启用 0禁用")
    is_superuser: bool = Field(default=False, description="是否超级管理员")
    role_ids: list[UUID] = Field(default_factory=list, description="角色 ID 列表")


class UserUpdate(ApiInModel):
    """更新用户的入参（账号与密码不在此修改，密码走单独的重置接口）。"""

    nickname: Optional[str] = Field(default=None, max_length=50, description="昵称")
    email: Optional[str] = Field(default=None, max_length=100, description="邮箱")
    phone: Optional[str] = Field(default=None, max_length=20, description="手机号")
    avatar: Optional[str] = Field(default=None, max_length=500, description="头像")
    status: int = Field(default=1, ge=0, le=1, description="1启用 0禁用")
    is_superuser: bool = Field(default=False, description="是否超级管理员")
    role_ids: Optional[list[UUID]] = Field(default=None, description="角色 ID 列表（None 表示不修改角色）")


class UserResetPassword(ApiInModel):
    """重置用户密码的入参。"""

    user_id: UUID = Field(description="用户 ID")
    password: str = Field(min_length=6, max_length=50, description="新密码")


class UserOut(ApiOutModel):
    """用户的出参（从 ORM 对象直接转换）。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    username: str
    email: Optional[str] = None
    phone: Optional[str] = None
    nickname: Optional[str] = None
    avatar: Optional[str] = None
    status: int
    is_superuser: bool
    last_login_ip: Optional[str] = None
    roles: list[RoleOut] = Field(default_factory=list)
    create_time: UtcDateTime
    update_time: UtcDateTime

    @field_serializer("id")
    def serialize_id(self, value: UUID) -> str:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex


class UserQuery(ApiInModel):
    """用户多条件查询（POST body）。"""

    page: int = Field(default=1, ge=1, description="页码（从 1 开始）")
    pageSize: int = Field(default=20, ge=10, le=100, description="每页条数")
    username: str | None = Field(default=None, max_length=50, description="关键字：用户名模糊匹配")
    nickname: str | None = Field(default=None, max_length=20, description="关键字：昵称模糊匹配")
    status: int | None = Field(default=None, ge=0, le=1, description="状态筛选：1启用 0禁用")
    role_id: UUID | None = Field(default=None, description="按角色筛选")
