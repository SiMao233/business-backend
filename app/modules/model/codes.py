"""模型管理权限码常量。

权限码为「模块:资源:动作」扁平标识，供接口鉴权（require_permissions）与前端按钮鉴权共用。
对应数据需在 sys_permission 表中维护（type=3 按钮），并分配角色后生效。
"""


class PermissionCode:
    """模型管理-权限码。"""

    # ---- 模型供应商 ----
    MODEL_PROVIDER_LIST = "model:provider:list"
    MODEL_PROVIDER_CREATE = "model:provider:create"
    MODEL_PROVIDER_UPDATE = "model:provider:update"
    MODEL_PROVIDER_DELETE = "model:provider:delete"

    # ---- 模型实例 ----
    MODEL_INSTANCE_LIST = "model:instance:list"
    MODEL_INSTANCE_CREATE = "model:instance:create"
    MODEL_INSTANCE_UPDATE = "model:instance:update"
    MODEL_INSTANCE_DELETE = "model:instance:delete"
