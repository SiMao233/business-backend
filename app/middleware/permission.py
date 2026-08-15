"""权限校验中间件与权限依赖。

接口级权限校验通过 `require_permissions(*codes)` 依赖实现：接口声明所需权限码，
请求进入时校验当前用户是否拥有（超级管理员自动放行），不满足抛 `PermissionDeniedError`(403)。
`PermissionMiddleware` 为全局中间件骨架，未注册——接口级权限用依赖注入更精确、可组合。
"""

from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.database import get_db
from app.core.exceptions import PermissionDeniedError
from app.middleware.authentication import UserContext, get_current_user


class PermissionMiddleware(BaseHTTPMiddleware):  # pragma: no cover - 骨架，暂未注册
    """HTTP 权限中间件骨架。

    当前直接放行请求；TODO(rbac): 校验 `request.state.user` 的权限后放行或拒绝。
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        return await call_next(request)


def require_permissions(*permissions: str):
    """生成一个校验指定权限的 FastAPI 依赖。

    用法::

        @router.post("/create", dependencies=[Depends(require_permissions("system:user:create"))])
        async def create_user(...): ...

    校验逻辑：先经 `get_current_user` 完成认证；超级管理员直接放行；普通用户聚合其
    所有启用角色绑定的权限码（含父链继承，拥有子权限即拥有祖先权限），与所需权限码
    求交集，为空则抛出 `PermissionDeniedError`(403)。
    """

    async def _dependency(
        user: Annotated[UserContext, Depends(get_current_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> UserContext:
        # 延迟导入：避免与 app.modules/system/permission/api 的循环依赖
        # （app.modules/__init__ eager 加载 system.api，顶层 import repository 会触发回环）
        from app.modules.system.permission.repository import PermissionRepository

        assert user.user_id is not None  # 认证通过后必有用户 ID
        repo = PermissionRepository(db)
        # 超级管理员自动放行，无需为其配置全部权限
        if await repo.is_superuser(user.user_id):
            return user
        owned = await repo.get_user_permission_codes(user.user_id)
        if not owned.intersection(permissions):
            raise PermissionDeniedError("没有操作权限")
        return user

    return _dependency
