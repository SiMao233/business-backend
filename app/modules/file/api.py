"""文件模块 API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。

本层禁止直接操作数据库或编写复杂业务逻辑。
"""

from typing import Annotated
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import FileBizType
from app.common.response import ApiResponse, success
from app.core.database import get_db
from app.middleware.authentication import CurrentUserDep, get_current_user
from app.modules.file.schema import FileOut
from app.modules.file.service import FileContent, FileService

# 注意：router 级依赖必须用 Depends(get_current_user)（函数式），
# 用 Depends(CurrentUserDep)（Annotated）会在 FastAPI 0.141 下破坏 multipart 表单解析（query.args 报错）。
router = APIRouter(prefix="/file", tags=["文件管理"], dependencies=[Depends(get_current_user)])

DbDep = Annotated[AsyncSession, Depends(get_db)]


# 依赖注入工厂：创建 FileService 实例（绑定数据库会话）
def get_service(db: DbDep) -> FileService:
    return FileService(db)


def _content_disposition(record_name: str, inline: bool) -> str:
    """构造 Content-Disposition 头。

    - ASCII 文件名直接放入 filename；非 ASCII 用 RFC 5987 filename* 编码，避免乱码。
    """
    disposition = "inline" if inline else "attachment"
    ascii_name = record_name.encode("ascii", "ignore").decode() or "file"
    quoted = ascii_name.replace('"', "")
    encoded = quote(record_name)
    return f'{disposition}; filename="{quoted}"; filename*=UTF-8\'\'{encoded}'


def _file_response(content: FileContent, inline: bool) -> Response:
    """根据存储后端返回流式文件响应（本地走 FileResponse，否则回退字节）。"""
    headers = {"Content-Disposition": _content_disposition(content.record.name, inline)}
    if content.path is not None:
        return FileResponse(content.path, media_type=content.record.content_type, headers=headers)
    return Response(content=content.data, media_type=content.record.content_type, headers=headers)


# 上传文件接口：multipart 表单，file 必填，bizType 可选
@router.post(
    "/upload",
    response_model=ApiResponse[FileOut],
    summary="上传文件",
)
async def upload_file(
    file: Annotated[UploadFile, File(description="文件内容")],
    biz_type: Annotated[FileBizType | None, Form()] = None,
    current_user: CurrentUserDep = None,
    service: Annotated[FileService, Depends(get_service)] = None,
) -> ApiResponse[FileOut]:
    return success(data=await service.upload(file, current_user.user_id, biz_type))


# 下载文件接口：attachment 方式返回
@router.get(
    "/{file_id}/download",
    summary="下载文件",
    response_class=Response,
)
async def download_file(
    file_id: UUID,
    service: Annotated[FileService, Depends(get_service)],
) -> Response:
    return _file_response(await service.get_content(file_id), inline=False)


# 预览文件接口：inline 方式返回（浏览器直接打开图片 / PDF 等）
@router.get(
    "/preview/{file_id}",
    summary="预览文件",
    response_class=Response,
)
async def preview_file(
    file_id: UUID,
    service: Annotated[FileService, Depends(get_service)],
) -> Response:
    return _file_response(await service.get_content(file_id), inline=True)
