"""系统管理模块模型聚合：集中导入系统模块内的 ORM 模型。

供 Alembic 迁移识别差异使用。
"""

from app.modules.system.permission.model import Permission
from app.modules.system.role.model import Role
from app.modules.system.user.model import User

__all__ = ["User", "Role", "Permission"]
