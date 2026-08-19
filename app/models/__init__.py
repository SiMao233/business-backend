# 用户
# 文件
from app.modules.file.model import SysFile

# 关联表
from app.modules.iam.model import RolePermission, UserRole

# 权限
from app.modules.system.permission.model import Permission

# 角色
from app.modules.system.role.model import Role
from app.modules.system.user.model import User

__all__ = [
    "User",
    "Role",
    "Permission",
    "UserRole",
    "RolePermission",
    "SysFile",
]
