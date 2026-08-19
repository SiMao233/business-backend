"""文件模块 Repository 层：数据访问（CRUD / 查询）。"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.repository import BaseRepository
from app.modules.file.model import SysFile


class FileRepository(BaseRepository):
    """文件元数据数据访问仓库。"""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, SysFile)

    # 根据主键 ID 查询有效文件（status=1）
    async def get_by_id(self, file_id: UUID) -> SysFile | None:
        stmt = select(SysFile).where(SysFile.id == file_id, SysFile.status == 1)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 根据存储 key 查询（唯一性校验 / 去重）
    async def get_by_storage_key(self, storage_key: str) -> SysFile | None:
        stmt = select(SysFile).where(SysFile.storage_key == storage_key)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 新增文件元数据：写入数据库并提交事务，刷新后返回对象
    async def create(self, file: SysFile) -> SysFile:
        self.db.add(file)
        await self.db.commit()
        await self.db.refresh(file)
        return file

    # 逻辑删除：置 status=0 并提交
    async def soft_delete(self, file: SysFile) -> None:
        file.status = 0
        await self.db.commit()
