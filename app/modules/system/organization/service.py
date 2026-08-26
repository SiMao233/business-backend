"""组织管理 Service 层：业务规则、权限判断、流程编排。"""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.common.pagination import PageParams, PageResult
from app.core.exceptions import BizError, NotFoundError
from app.modules.system.organization.model import Organization
from app.modules.system.organization.repository import OrganizationRepository
from app.modules.system.organization.schema import (
    OrganizationCreate,
    OrganizationNode,
    OrganizationOut,
    OrganizationQuery,
    OrganizationUpdate,
)


class OrganizationService:
    """组织管理服务。"""

    def __init__(self, db: AsyncSession) -> None:
        self.repo = OrganizationRepository(db)

    # 分页查询组织列表
    async def list_page(self, query: OrganizationQuery) -> PageResult[OrganizationOut]:
        page_params = PageParams(page=query.page, pageSize=query.pageSize)
        page = await self.repo.list_page(
            page_params, keyword=query.keyword, status=query.status
        )
        return PageResult(
            list=[OrganizationOut.model_validate(item) for item in page.list],
            total=page.total,
            page=page.page,
            pageSize=page.pageSize,
        )

    # 查询单个组织详情
    async def get(self, org_id: UUID) -> OrganizationOut:
        org = await self.repo.get_by_id(org_id)
        if not org:
            raise NotFoundError("组织不存在")
        return OrganizationOut.model_validate(org)

    # 创建组织：校验编码唯一性与上级组织有效性
    async def create(self, req: OrganizationCreate) -> OrganizationOut:
        if await self.repo.get_by_code(req.code):
            raise BizError("组织编码已存在")
        if req.parent_id:
            parent = await self.repo.get_by_id(req.parent_id)
            if not parent:
                raise NotFoundError("上级组织不存在")
        org = Organization(
            name=req.name,
            code=req.code,
            description=req.description,
            parent_id=req.parent_id,
            owner_id=req.owner_id,
            status=req.status,
        )
        return OrganizationOut.model_validate(await self.repo.create(org))

    # 更新组织：校验上级组织不能是自身
    async def update(self, org_id: UUID, req: OrganizationUpdate) -> OrganizationOut:
        org = await self.repo.get_by_id(org_id)
        if not org:
            raise NotFoundError("组织不存在")
        if req.parent_id is not None and req.parent_id == org_id:
            raise BizError("上级组织不能是自身")
        if req.name is not None:
            org.name = req.name
        if req.description is not None:
            org.description = req.description
        if req.parent_id is not None:
            org.parent_id = req.parent_id
        if req.owner_id is not None:
            org.owner_id = req.owner_id
        if req.status is not None:
            org.status = req.status
        return OrganizationOut.model_validate(await self.repo.update(org))

    # 删除组织
    async def delete(self, org_id: UUID) -> None:
        org = await self.repo.get_by_id(org_id)
        if not org:
            raise NotFoundError("组织不存在")
        await self.repo.delete(org)

    # 组织树：全量加载后内存组装（不依赖 ORM children 自动填充，避免重复）
    async def tree(self) -> list[OrganizationNode]:
        orgs = await self.repo.list_all()
        nodes: dict[UUID, OrganizationNode] = {}
        for org in orgs:
            nodes[org.id] = OrganizationNode(
                id=org.id,
                parent_id=org.parent_id,
                name=org.name,
                code=org.code,
                description=org.description,
                owner_id=org.owner_id,
                status=org.status,
            )
        roots: list[OrganizationNode] = []
        for org in orgs:
            node = nodes[org.id]
            if org.parent_id is not None and org.parent_id in nodes:
                nodes[org.parent_id].children.append(node)
            else:
                roots.append(node)
        return roots
