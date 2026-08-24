"""数据库初始化脚本（幂等 seed）。

用法：
    python -m app.core.seed

在（空）数据库上初始化系统基础数据：
1. 内置权限树：目录「系统管理」→ 菜单「用户/角色/权限管理」→ 按钮（权限码，来自
   ``app.modules.system.permission.codes.PermissionCode``），全部标记 ``is_builtin=True``；
2. 内置角色「超级管理员」（``code=admin``，``is_builtin=True``），绑定全部内置权限；
3. 初始超管账号 ``admin``（``is_superuser=True``），绑定内置角色，初始密码取配置
   ``app_admin_password``（可用环境变量 ``APP_ADMIN_PASSWORD`` 覆盖）。

脚本幂等：重复执行不会重复插入，也不会覆盖运营后续对数据的修改（角色/用户权限为并集合并）。
"""

import asyncio

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal, dispose_engine
from app.core.security import hash_password
from app.models import Permission, Role, User
from app.modules.system.permission.codes import PermissionCode

# 内置角色与初始超管
BUILTIN_ROLE_CODE = "admin"
BUILTIN_ROLE_NAME = "超级管理员"
ADMIN_USERNAME = "admin"
ADMIN_NICKNAME = "超级管理员"


def _btn(code: str, name: str) -> dict:
    """构造一个按钮节点（type=3）。"""
    return {"code": code, "name": name, "type": 3}


# 内置权限树（目录 -> 菜单 -> 按钮）；按钮 code 一律引用 PermissionCode，保持单一事实来源
PERMISSION_TREE: list[dict] = [
    {
        "name": "系统管理",
        "code": "system",
        "type": 1,
        "children": [
            {
                "name": "用户管理",
                "code": "system:user",
                "type": 2,
                "children": [
                    _btn(PermissionCode.USER_LIST, "用户列表"),
                    _btn(PermissionCode.USER_CREATE, "创建用户"),
                    _btn(PermissionCode.USER_UPDATE, "更新用户"),
                    _btn(PermissionCode.USER_RESET_PASSWORD, "重置密码"),
                    _btn(PermissionCode.USER_DELETE, "删除用户"),
                ],
            },
            {
                "name": "角色管理",
                "code": "system:role",
                "type": 2,
                "children": [
                    _btn(PermissionCode.ROLE_LIST, "角色列表"),
                    _btn(PermissionCode.ROLE_CREATE, "创建角色"),
                    _btn(PermissionCode.ROLE_UPDATE, "更新角色"),
                    _btn(PermissionCode.ROLE_ASSIGN_PERMISSION, "分配权限"),
                    _btn(PermissionCode.ROLE_DELETE, "删除角色"),
                ],
            },
            {
                "name": "权限管理",
                "code": "system:permission",
                "type": 2,
                "children": [
                    _btn(PermissionCode.PERMISSION_LIST, "权限列表"),
                    _btn(PermissionCode.PERMISSION_CREATE, "创建权限"),
                    _btn(PermissionCode.PERMISSION_UPDATE, "更新权限"),
                    _btn(PermissionCode.PERMISSION_DELETE, "删除权限"),
                ],
            },
            {
                "name": "操作日志",
                "code": "system:operationLog",
                "type": 2,
                "children": [
                    _btn(PermissionCode.OPERATION_LOG_LIST, "操作日志列表"),
                ],
            },
        ],
    },
]


async def _sync_node(
    db: AsyncSession,
    existing: dict[str, Permission],
    node: dict,
    parent: Permission | None,
    all_perms: list[Permission],
    created: list[str],
) -> Permission:
    """幂等同步单个权限节点（目录/菜单/按钮）。"""
    code: str = node["code"]
    perm = existing.get(code)
    if perm is None:
        perm = Permission(
            code=code,
            name=node["name"],
            type=node["type"],
            parent_id=parent.id if parent else None,
            status=1,
            is_builtin=True,
        )
        db.add(perm)
        await db.flush()  # 获取主键 UUID，供子节点挂 parent_id
        created.append(code)
    else:
        # 已存在：同步名称/类型/父节点与定义一致，但尊重运营对 status 的调整
        perm.name = node["name"]
        perm.type = node["type"]
        perm.parent_id = parent.id if parent else None
    all_perms.append(perm)
    for child in node.get("children", []):
        await _sync_node(db, existing, child, perm, all_perms, created)
    return perm


async def sync_permission_tree(db: AsyncSession) -> list[Permission]:
    """同步内置权限树，返回全部内置权限（目录/菜单/按钮）。"""
    existing = {p.code: p for p in (await db.execute(select(Permission))).scalars()}
    all_perms: list[Permission] = []
    created: list[str] = []
    for root in PERMISSION_TREE:
        await _sync_node(db, existing, root, None, all_perms, created)
    if created:
        print(f"  新增内置权限 {len(created)} 个：{', '.join(created)}")
    else:
        print("  内置权限已齐全，无需新增")
    return all_perms


async def ensure_builtin_role(db: AsyncSession, perms: list[Permission]) -> Role:
    """确保内置角色存在并绑定全部内置权限（并集合并，不撤销既有分配）。"""
    stmt = (
        select(Role)
        .options(selectinload(Role.permissions))
        .where(Role.code == BUILTIN_ROLE_CODE)
    )
    role = await db.scalar(stmt)
    if role is None:
        # 构造时直接绑定全部内置权限，避免 flush 后再赋值触发异步环境下的惰性加载
        role = Role(
            code=BUILTIN_ROLE_CODE,
            name=BUILTIN_ROLE_NAME,
            status=1,
            is_builtin=True,
            permissions=list(perms),
        )
        db.add(role)
        await db.flush()
        print(f"  新增内置角色：{BUILTIN_ROLE_NAME}（{BUILTIN_ROLE_CODE}）")
    else:
        role.is_builtin = True  # 历史数据兜底标记
        role.permissions = list({p.id: p for p in [*role.permissions, *perms]}.values())
    await db.flush()
    return role


async def ensure_admin(db: AsyncSession, role: Role) -> User:
    """确保系统存在名为 ``admin`` 的超管账号并绑定内置角色。

    若 ``admin`` 不存在则创建（密码取配置 ``app_admin_password``）；
    若已存在（例如历史手动创建的账号），则将其提升为超管并绑定内置角色，
    以保证"初始超管账号"语义始终成立。
    """
    stmt = (
        select(User)
        .options(selectinload(User.roles))
        .where(User.username == ADMIN_USERNAME)
    )
    user = await db.scalar(stmt)
    if user is None:
        password = get_settings().app_admin_password
        user = User(
            username=ADMIN_USERNAME,
            password_hash=hash_password(password),
            nickname=ADMIN_NICKNAME,
            status=1,
            is_superuser=True,
            roles=[role],
        )
        db.add(user)
        await db.flush()
        print(f"  新增初始超管账号：{ADMIN_USERNAME}（初始密码取自配置 app_admin_password，请尽快修改）")
    else:
        # 已存在：确保超管身份与内置角色绑定
        user.is_superuser = True
        user.roles = list({r.id: r for r in [*user.roles, role]}.values())
    await db.flush()
    return user


async def seed() -> None:
    """执行完整初始化（幂等）。"""
    async with AsyncSessionLocal() as db:
        async with db.begin():
            perms = await sync_permission_tree(db)
            role = await ensure_builtin_role(db, perms)
            user = await ensure_admin(db, role)
        logger.info("初始化超管完成 username={}", user.username)
        print(
            f"Seed 完成：内置权限 {len(perms)} 个，角色「{role.name}」，"
            f"账号「{user.username}」（is_superuser={user.is_superuser}）"
        )


async def main() -> None:
    """脚本入口：执行 seed 后释放数据库连接池。"""
    print("=== 开始初始化系统基础数据 ===")
    try:
        await seed()
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
