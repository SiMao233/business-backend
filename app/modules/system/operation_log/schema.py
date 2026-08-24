"""系统操作日志 Pydantic Schema：请求 / 响应模型。"""

from uuid import UUID

from pydantic import ConfigDict, Field, field_serializer

from app.common.schema import ApiInModel, ApiOutModel, UtcDateTime


class OperationLogOut(ApiOutModel):
    """操作日志出参（从 ORM 对象直接转换）。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID | None = None
    username: str | None = None
    module: str
    action: str
    method: str | None = None
    path: str | None = None
    target_id: UUID | None = None
    detail: str | None = None
    ip: str | None = None
    status: int
    error_msg: str | None = None
    create_time: UtcDateTime

    @field_serializer("id")
    def serialize_id(self, value: UUID) -> str:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex

    @field_serializer("user_id", "target_id")
    def serialize_optional_uuid(self, value: UUID | None) -> str | None:
        """可空 UUID 同样输出 32 位十六进制字符串。"""
        return value.hex if value else None


class OperationLogQuery(ApiInModel):
    """操作日志多条件查询（POST body）。"""

    page: int = Field(default=1, ge=1, description="页码（从 1 开始）")
    pageSize: int = Field(default=20, ge=10, le=100, description="每页条数")
    module: str | None = Field(default=None, max_length=50, description="模块筛选：user/role/permission 等")
    action: str | None = Field(default=None, max_length=50, description="动作筛选：delete/assignPermissions 等")
    username: str | None = Field(default=None, max_length=50, description="操作人用户名模糊匹配")
    status: int | None = Field(default=None, ge=0, le=1, description="结果筛选：1成功 0失败")