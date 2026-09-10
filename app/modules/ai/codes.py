"""AI 能力权限码常量。

权限码为「模块:资源:动作」扁平标识，供接口鉴权（require_permissions）与前端按钮鉴权共用。
对应数据需在 sys_permission 表中维护（type=3 按钮），并分配角色后生效。
"""


class PermissionCode:
    """AI 能力-权限码。"""

    AI_CHAT = "ai:chat:chat"
    CONVERSATION_LIST = "ai:conversation:list"
    CONVERSATION_CREATE = "ai:conversation:create"
    CONVERSATION_DELETE = "ai:conversation:delete"
