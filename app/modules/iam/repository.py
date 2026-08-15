"""IAM Repository 层：数据访问（认证相关查询）。

复用 `app.modules.system.user.model.User`（表 `sys_user`），避免重复建表。
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.common.repository import BaseRepository
from app.modules.system.user.model import User


class IamRepository(BaseRepository[User]):
    """用户数据访问仓库（认证场景）。"""

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(db, User)

    # 根据用户名查询用户（登录）
    async def get_by_username(self, username: str) -> User | None:
        stmt = select(User).where(User.username == username)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 根据主键 ID 查询用户（预加载角色关系，避免异步懒加载报错）
    async def get_by_id(self, user_id: UUID) -> User | None:
        stmt = (
            select(User)
            .options(selectinload(User.roles))
            .where(User.id == user_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # 登录成功后更新最近登录 IP 并提交
    async def update_login_info(self, user: User, ip: str) -> None:
        user.last_login_ip = ip
        await self.db.commit()
