"""Agent 权限码常量。

权限码为「模块:资源:动作」扁平标识，供接口鉴权（require_permissions）与前端按钮鉴权共用。
对应数据需在 sys_permission 表中维护（type=3 按钮），并分配角色后生效。
"""


class PermissionCode:
    """Agent 管理-权限码。"""

    AGENT_LIST = "agent:agent:list"
    AGENT_CREATE = "agent:agent:create"
    AGENT_UPDATE = "agent:agent:update"
    AGENT_DELETE = "agent:agent:delete"
    AGENT_PUBLISH = "agent:agent:publish"
    AGENT_RUN = "agent:agent:run"
    AGENT_VERSION_LIST = "agent:agent:versionList"
