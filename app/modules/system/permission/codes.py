"""系统权限码常量。

权限码为「模块:资源:动作」的扁平标识，供接口鉴权（`require_permissions`）与前端按钮鉴权共用。
对应数据需在 `sys_permission` 表中维护（type=3 按钮），并分配给角色后方可生效。
"""


class PermissionCode:
    """系统管理-权限码。"""

    # ---- 系统-用户 ----
    USER_LIST = "system:user:list"
    USER_CREATE = "system:user:create"
    USER_UPDATE = "system:user:update"
    USER_RESET_PASSWORD = "system:user:resetPassword"
    USER_DELETE = "system:user:delete"

    # ---- 系统-角色 ----
    ROLE_LIST = "system:role:list"
    ROLE_CREATE = "system:role:create"
    ROLE_UPDATE = "system:role:update"
    ROLE_ASSIGN_PERMISSION = "system:role:assignPermission"
    ROLE_DELETE = "system:role:delete"

    # ---- 系统-权限 ----
    PERMISSION_LIST = "system:permission:list"
    PERMISSION_CREATE = "system:permission:create"
    PERMISSION_UPDATE = "system:permission:update"
    PERMISSION_DELETE = "system:permission:delete"
