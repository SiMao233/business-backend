"""用量统计权限码常量。

权限码为「模块:资源:动作」扁平标识，供接口鉴权（require_permissions）与前端按钮鉴权共用。
对应数据需在 sys_permission 表中维护（type=3 按钮），并分配角色后生效。
"""


class PermissionCode:
    """用量统计-权限码。"""

    # 查看用量明细（管理端）
    USAGE_RECORD_LIST = "usage:record:list"
    # 查看统计 / 趋势 / 概览（管理端）
    USAGE_STAT_VIEW = "usage:stat:view"
    # 数据范围：可查看全局用量（无此码仅能查看自己的数据）
    USAGE_STAT_ALL = "usage:stat:all"
