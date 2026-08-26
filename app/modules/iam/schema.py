"""IAM Pydantic Schema：认证请求 / 响应模型。"""

from uuid import UUID

from pydantic import ConfigDict, Field, field_serializer

from app.common.schema import ApiInModel, ApiOutModel, UtcDateTime


class LoginRequest(ApiInModel):
    """登录入参。"""

    username: str = Field(min_length=1, max_length=50, description="登录账号")
    password: str = Field(min_length=1, max_length=512, description="密码")


class RefreshRequest(ApiInModel):
    """刷新令牌入参。"""

    refresh_token: str = Field(min_length=1, description="刷新令牌")


class UserInfoOut(ApiOutModel):
    """当前登录用户信息出参。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    username: str
    nickname: str | None = None
    email: str | None = None
    avatar: str | None = None
    is_superuser: bool
    status: int
    last_login_ip: str | None = None
    create_time: UtcDateTime
    update_time: UtcDateTime

    @field_serializer("id")
    def serialize_id(self, value: UUID) -> str:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex


class TokenOut(ApiOutModel):
    """登录 / 刷新成功后的令牌响应。"""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserInfoOut
