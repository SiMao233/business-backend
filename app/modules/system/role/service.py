"""系统角色管理 Service 层：业务规则、权限判断、流程编排。
"""

from uuid import UUID

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageParams, PageResult
from app.core.exceptions import BizError, NotFoundError
from app.middleware.authentication import UserContext
from app.modules.system.permission.repository import PermissionRepository
from app.modules.system.role.model import Role
from app.modules.system.role.repository import RoleRepository
from app.modules.system.role.schema import (
    RoleCreate,
    RoleOut,
    RolePermissionsReq,
    RoleQuery,
    RoleUpdate,
)


class RoleService:
    """后台角色管理服务。"""

    def __init__(self, db: AsyncSession) -> None:
        self.repo = RoleRepository(db)

    async def list_page(self, query: RoleQuery) -> PageResult[RoleOut]:
        pageParams = PageParams(page=query.page, pageSize=query.pageSize)
        page = await self.repo.list_page(
            pageParams,
            name=query.name,
            code=query.code,
            status=query.status,
        )
        return PageResult(
            list=[RoleOut.model_validate(item) for item in page.list],
            total=page.total,
            page=page.page,
            pageSize=page.pageSize,
        )

    async def get(self, role_id: UUID) -> RoleOut:
        role = await self.repo.get_by_id(role_id)
        if not role:
            raise NotFoundError("角色不存在")
        return RoleOut.model_validate(role)

    async def create(self, req: RoleCreate) -> RoleOut:
        if await self.repo.get_by_code(req.code):
            raise BizError("角色编码已存在")
        role = Role(
            code=req.code,
            name=req.name,
            description=req.description,
            status=req.status,
        )
        return RoleOut.model_validate(await self.repo.create(role))

    async def update(self, role_id: UUID, req: RoleUpdate, operator: UserContext | None = None) -> RoleOut:
        role = await self.repo.get_by_id(role_id)
        if not role:
            raise NotFoundError("角色不存在")
        # 系统内置角色：禁止禁用（仅允许改名称/描述）
        if role.is_builtin and req.status == 0:
            raise BizError("系统内置角色不允许禁用")
        role.name = req.name
        role.description = req.description
        role.status = req.status
        logger.info("更新角色 role_id={} by={}", role_id, operator.username if operator else "system")
        return RoleOut.model_validate(await self.repo.update(role))

    async def delete(self, role_id: UUID, operator: UserContext | None = None) -> Role:
        role = await self.repo.get_by_id(role_id)
        if not role:
            raise NotFoundError("角色不存在")
        if role.is_builtin:
            raise BizError("系统内置角色不允许删除")
        await self.repo.delete(role)
        logger.warning(
            "删除角色 role_id={} code={} by={}",
            role_id, role.code, operator.username if operator else "system",
        )
        return role

    # 分配权限（全量覆盖）：校验权限存在性并自动补全父链节点后绑定
    async def assign_permissions(self, role_id: UUID, req: RolePermissionsReq, operator: UserContext | None = None) -> RoleOut:
        role = await self.repo.get_by_id(role_id)
        if not role:
            raise NotFoundError("角色不存在")
        ids = list(dict.fromkeys(req.permission_ids))
        permissions = await self.repo.get_permissions_by_ids(ids)
        if len(permissions) != len(ids):
            raise BizError("存在无效的权限 ID")
        all_perms = await self.repo.list_all_permissions()
        final_ids = PermissionRepository.expand_with_ancestors(all_perms, set(ids))
        final_perms = [p for p in all_perms if p.id in final_ids]
        await self.repo.set_permissions(role, final_perms)
        logger.info(
            "角色分配权限 role_id={} 权限数={} by={}",
            role_id, len(final_perms), operator.username if operator else "system",
        )
        return RoleOut.model_validate(role)
