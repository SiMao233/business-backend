"""文件模块 Service 层：上传校验、存储编排、元数据落库、下载内容获取。

本层不直接操作 HTTP 响应，仅返回业务结果；存储后端通过 `StorageBackend` 抽象注入，
当前默认使用本地磁盘实现（`LocalStorage`）。
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.enums import FileBizType
from app.core.config import get_settings
from app.core.exceptions import NotFoundError, ValidateError
from app.modules.file.model import SysFile
from app.modules.file.repository import FileRepository
from app.modules.file.schema import FileOut
from app.modules.file.storage import LocalStorage, StorageBackend, generate_storage_key

settings = get_settings()

# 分块读取大小（1MB）
_CHUNK_SIZE = 1024 * 1024


@dataclass
class FileContent:
    """下载/预览内容载体：本地存储走文件路径（FileResponse 流式），否则走字节。"""

    record: SysFile
    path: Path | None = None
    data: bytes | None = None


def _extract_ext(filename: str | None) -> str | None:
    """从原始文件名提取扩展名（不含点，小写）；无扩展名返回 None。"""
    if not filename:
        return None
    name = filename.rsplit("/", 1)[-1]  # 兼容部分客户端携带路径
    if "." not in name:
        return None
    return name.rsplit(".", 1)[-1].lower()


async def _read_limited(file: UploadFile, max_size: int) -> bytes:
    """分块读取上传内容，超过大小上限抛 ValidateError。"""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_size:
            raise ValidateError(f"文件大小超过限制（最大 {max_size // (1024 * 1024)}MB）")
        chunks.append(chunk)
    return b"".join(chunks)


class FileService:
    """文件上传 / 下载服务。"""

    def __init__(self, db: AsyncSession, storage: StorageBackend | None = None) -> None:
        self.repo = FileRepository(db)
        self.storage = storage or LocalStorage()

    # 上传：校验 → 保存存储 → 落元数据 → 返回出参（含下载地址）
    async def upload(
        self,
        file: UploadFile,
        uploader_id: UUID | None,
        biz_type: FileBizType | None = None,
    ) -> FileOut:
        ext = _extract_ext(file.filename)
        if ext and ext not in settings.file_allowed_extensions:
            raise ValidateError(f"不支持的文件类型: {ext}")

        data = await _read_limited(file, settings.file_max_size)
        if not data:
            raise ValidateError("文件内容为空")

        storage_key = generate_storage_key(ext)
        await self.storage.save(storage_key, data)

        record = SysFile(
            name=file.filename or "unnamed",
            storage_key=storage_key,
            content_type=file.content_type or "application/octet-stream",
            size=len(data),
            ext=ext,
            sha256=hashlib.sha256(data).hexdigest(),
            uploader_id=uploader_id,
            biz_type=biz_type.value if biz_type else None,
        )
        record = await self.repo.create(record)

        out = FileOut.model_validate(record)
        out.url = self._build_url(record.id)
        return out

    # 获取下载/预览内容：查元数据（不存在抛 NotFoundError），返回本地路径或字节
    async def get_content(self, file_id: UUID) -> FileContent:
        record = await self.repo.get_by_id(file_id)
        if not record:
            raise NotFoundError("文件不存在")
        path = self.storage.get_path(record.storage_key)
        if path is not None:
            return FileContent(record=record, path=path)
        try:
            data = await self.storage.open(record.storage_key)
        except FileNotFoundError:
            raise NotFoundError("文件内容不存在") from None
        return FileContent(record=record, data=data)

    # 构造下载地址：配置了 file_public_base_url 则拼接完整 URL，否则返回相对路径
    def _build_url(self, file_id: UUID) -> str:
        path = f"/api/v1/file/{file_id.hex}/download"
        base = settings.file_public_base_url.rstrip("/")
        return f"{base}{path}" if base else path
