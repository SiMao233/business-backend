"""系统权限管理 Service 层：业务规则、权限判断、流程编排。"""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageParams, PageResult
from app.core.exceptions import BizError, NotFoundError
from app.modules.system.permission.model import Permission
from app.modules.system.permission.repository import PermissionRepository
from app.modules.system.permission.schema import (
    PermissionCreate,
    PermissionNode,
    PermissionOut,
    PermissionQuery,
    PermissionUpdate,
    UserPermissionOut,
)


class PermissionService:
    """后台权限管理服务。"""

    def __init__(self, db: AsyncSession) -> None:
        self.repo = PermissionRepository(db)

    # 分页查询权限列表，支持关键字 / 类型 / 状态筛选
    async def list_page(self, query: PermissionQuery) -> PageResult[PermissionOut]:
        page_params = PageParams(page=query.page, pageSize=query.pageSize)
        page = await self.repo.list_page(
            page_params,
            keyword=query.keyword,
            type_=query.type,
            status=query.status,
        )
        items = [self._to_out(p) for p in page.list]
        return PageResult(
            list=items,
            total=page.total,
            page=page.page,
            pageSize=page.pageSize,
        )

    # 查询单个权限详情
    async def get(self, permission_id: UUID) -> PermissionOut:
        permission = await self.repo.get_by_id(permission_id)
        if not permission:
            raise NotFoundError("权限不存在")
        return self._to_out(permission)

    # 权限树：全量权限按 parent_id 组装为嵌套树（根节点 = parent_id 为空的节点）
    async def tree(self) -> list[PermissionNode]:
        permissions = await self.repo.list_all()
        return self.build_tree(permissions)

    # 当前用户聚合权限：其所有启用角色绑定的权限码去重 + 父链补全，返回权限码集合与菜单树
    async def mine(self, user_id: UUID) -> UserPermissionOut:
        permission_ids = await self.repo.get_permission_ids_by_user(user_id)
        if not permission_ids:
            return UserPermissionOut()
        all_perms = await self.repo.list_all()
        final_ids = self.repo.expand_with_ancestors(all_perms, permission_ids)
        final_perms = [p for p in all_perms if p.id in final_ids]
        codes = [p.code for p in final_perms if p.status == 1]
        tree = self.build_tree(final_perms)
        return UserPermissionOut(codes=codes, tree=tree)

    # 创建权限：校验权限码唯一与父节点合法性
    async def create(self, req: PermissionCreate) -> PermissionOut:
        if await self.repo.get_by_code(req.code):
            raise BizError("权限码已存在")
        await self._validate_parent(req.parent_id)
        permission = Permission(
            code=req.code,
            name=req.name,
            type=req.type,
            parent_id=req.parent_id,
            description=req.description,
            status=req.status,
        )
        return self._to_out(await self.repo.create(permission))

    # 更新权限：校验权限码唯一、父节点合法性（自身及其子孙不可作为父，防环）
    async def update(self, permission_id: UUID, req: PermissionUpdate) -> PermissionOut:
        permission = await self.repo.get_by_id(permission_id)
        if not permission:
            raise NotFoundError("权限不存在")
        await self._validate_parent(req.parent_id, exclude_id=permission_id)
        permission.name = req.name
        permission.type = req.type
        permission.parent_id = req.parent_id
        permission.description = req.description
        permission.status = req.status
        return self._to_out(await self.repo.update(permission))

    # 删除权限：存在子权限或被角色绑定时拒绝删除
    async def delete(self, permission_id: UUID) -> None:
        permission = await self.repo.get_by_id(permission_id)
        if not permission:
            raise NotFoundError("权限不存在")
        all_perms = await self.repo.list_all()
        if any(p.parent_id == permission_id for p in all_perms):
            raise BizError("存在子权限，请先删除子权限")
        if permission.roles:
            raise BizError("该权限已被角色绑定，请先解除绑定")
        await self.repo.delete(permission)

    # ---- 内部工具 ----

    # 权限点 → 出参（补充被角色引用的数量）
    @staticmethod
    def _to_out(permission: Permission) -> PermissionOut:
        out = PermissionOut.model_validate(permission)
        out.role_count = len(permission.roles)
        return out

    # 扁平权限列表 → 嵌套树（根节点 = parent_id 为 None，按 type、name 排序）
    @staticmethod
    def build_tree(permissions: list[Permission]) -> list[PermissionNode]:
        nodes: dict[UUID, PermissionNode] = {}
        for p in permissions:
            node = PermissionNode.model_validate(p)
            node.children = []
            nodes[p.id] = node
        roots: list[PermissionNode] = []
        for p in permissions:
            node = nodes[p.id]
            parent = nodes.get(p.parent_id) if p.parent_id else None
            if parent is not None:
                parent.children.append(node)
            else:
                roots.append(node)

        def sort_rec(items: list[PermissionNode]) -> None:
            items.sort(key=lambda n: (n.type, n.name))
            for child in items:
                sort_rec(child.children)

        sort_rec(roots)
        return roots

    # 校验父节点合法：存在、非自身、非自身子孙（防环）
    async def _validate_parent(self, parent_id: UUID | None, exclude_id: UUID | None = None) -> None:
        if parent_id is None:
            return
        if parent_id == exclude_id:
            raise BizError("上级权限不能是自身")
        parent = await self.repo.get_by_id(parent_id)
        if not parent:
            raise BizError("上级权限不存在")
        if exclude_id is not None:
            all_perms = await self.repo.list_all()
            subtree = self._collect_subtree_ids(all_perms, exclude_id)
            if parent_id in subtree:
                raise BizError("上级权限不能是其自身的子权限")

    # 收集某节点及其所有子孙的 ID（防环校验用）
    @staticmethod
    def _collect_subtree_ids(permissions: list[Permission], root_id: UUID) -> set[UUID]:
        result: set[UUID] = set()
        stack = [root_id]
        while stack:
            current = stack.pop()
            result.add(current)
            for p in permissions:
                if p.parent_id == current and p.id not in result:
                    stack.append(p.id)
        return result
