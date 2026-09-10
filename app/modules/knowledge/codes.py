"""知识库权限码常量。

权限码为「模块:资源:动作」扁平标识，供接口鉴权（require_permissions）与前端按钮鉴权共用。
对应数据需在 sys_permission 表中维护（type=3 按钮），并分配角色后生效。
"""


class PermissionCode:
    """知识库管理-权限码。"""

    KNOWLEDGE_LIST = "knowledge:knowledge:list"
    KNOWLEDGE_CREATE = "knowledge:knowledge:create"
    KNOWLEDGE_UPDATE = "knowledge:knowledge:update"
    KNOWLEDGE_DELETE = "knowledge:knowledge:delete"
    KNOWLEDGE_DOCUMENT_LIST = "knowledge:knowledge:documentList"
    KNOWLEDGE_DOCUMENT_UPLOAD = "knowledge:knowledge:documentUpload"
    KNOWLEDGE_DOCUMENT_DELETE = "knowledge:knowledge:documentDelete"
