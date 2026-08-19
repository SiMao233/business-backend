"""文件模块 Pydantic Schema：请求 / 响应模型。"""

from uuid import UUID

from pydantic import ConfigDict, Field, field_serializer

from app.common.schema import ApiOutModel, UtcDateTime


class FileOut(ApiOutModel):
    """文件元数据出参（从 ORM 对象转换，url 由 Service 计算填充）。"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str = Field(description="原始文件名")
    size: int = Field(description="文件大小（字节）")
    content_type: str = Field(description="MIME 类型")
    ext: str | None = Field(default=None, description="扩展名（不含点）")
    sha256: str | None = Field(default=None, description="内容 SHA-256")
    biz_type: str | None = Field(default=None, description="业务类型（FileBizType 枚举值）")
    uploader_id: UUID | None = Field(default=None, description="上传者 ID")
    status: int = Field(default=1, description="1 有效 / 0 逻辑删除")
    url: str | None = Field(default=None, description="下载地址（Service 计算填充）")
    create_time: UtcDateTime
    update_time: UtcDateTime

    @field_serializer("id")
    def serialize_id(self, value: UUID) -> str:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex

    @field_serializer("uploader_id")
    def serialize_uploader_id(self, value: UUID | None) -> str | None:
        """输出不带连字符的 32 位十六进制字符串。"""
        return value.hex if value else None
