"""文件存储抽象与本地实现。

业务层只依赖 `StorageBackend` 抽象接口；当前提供 `LocalStorage`（本地磁盘）实现，
后续如需切换 S3 / MinIO 等对象存储，新增实现类即可无缝替换，无需改动 Service 层。

存储 key 约定：`YYYY/MM/{uuid}.{ext}`（由服务端生成，杜绝用户输入导致的路径穿越）。
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.core.config import get_settings


class StorageBackend(ABC):
    """文件存储后端抽象接口。"""

    @abstractmethod
    async def save(self, key: str, data: bytes) -> None:
        """将文件内容写入存储，key 为存储路径标识。"""

    @abstractmethod
    async def open(self, key: str) -> bytes:
        """读取文件内容，不存在时抛 FileNotFoundError。"""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """删除文件，不存在时静默忽略。"""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """判断文件是否存在。"""

    def get_path(self, key: str) -> Path | None:
        """返回本地文件系统路径（仅本地存储实现可用），非本地返回 None。

        供下载/预览使用 `FileResponse` 流式返回（支持 Range 请求）；对象存储实现返回 None，
        由 Service 层回退为 `open()` 读取字节后返回。
        """
        return None


class LocalStorage(StorageBackend):
    """本地磁盘存储实现。

    根目录由配置 `file_storage_dir` 指定（相对项目根）。
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or get_settings().file_storage_dir)

    def _resolve(self, key: str) -> Path:
        # key 由服务端生成，此处仍做防御性校验，防止意外路径穿越
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"非法存储路径: {key}")
        return path

    async def save(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 写盘为阻塞 IO，放到线程池执行，避免阻塞事件循环
        await asyncio.to_thread(path.write_bytes, data)

    async def open(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return await asyncio.to_thread(path.read_bytes)

    async def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.is_file():
            await asyncio.to_thread(path.unlink)

    async def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def get_path(self, key: str) -> Path | None:
        path = self._resolve(key)
        return path if path.is_file() else None


def generate_storage_key(ext: str | None) -> str:
    """生成存储 key：`YYYY/MM/{uuid}.{ext}`（无扩展名则省略后缀）。"""
    now = datetime.now(UTC)
    name = uuid4().hex
    if ext:
        return f"{now:%Y/%m}/{name}.{ext}"
    return f"{now:%Y/%m}/{name}"
