"""模型聚合模块：集中导入所有 ORM 模型，供 Alembic 迁移识别差异。

新增模型后在此追加导入（确保模型注册进 Base.metadata）。
"""

from app.modules.iam.model import RolePermission, UserRole
from app.modules.system.models import Permission, Role, User


__all__ = ["User", "Role", "Permission", "UserRole", "RolePermission"]
