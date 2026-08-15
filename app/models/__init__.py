# 用户
from app.modules.system.user.model import User

# 角色
from app.modules.system.role.model import Role

# 权限
from app.modules.system.permission.model import Permission

# 关联表
from app.modules.iam.model import (
    UserRole,
    RolePermission
)

__all__ = [
    "User",
    "Role",
    "Permission",
    "UserRole",
    "RolePermission",
]