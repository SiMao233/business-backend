"""通用数据访问基类（Repository 层基础）。

各业务模块的 repository 继承 `BaseRepository`，统一会话注入与基础能力，
避免在 Service 层直接拼接 SQL。
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import Base


class BaseRepository[T: Base]:
    """所有业务 Repository 的基类。

    后续在此补充通用 CRUD（add / get / update / delete / list）等方法；
    各模块仓库继承后指定具体模型类型并添加领域查询方法。
    """

    def __init__(self, db: AsyncSession, model: type[T]) -> None:
        self.db = db
        self.model = model
