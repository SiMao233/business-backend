"""IAM API 层：HTTP 请求处理、参数校验、调用 Service、返回响应。

本层禁止直接操作数据库或编写复杂业务逻辑。
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.common.response import ApiResponse, success
from app.core.database import get_db
from app.core.exceptions import UnauthorizedError
from app.core.redis import get_redis
from app.core.security import load_public_key_pem, rsa_decrypt_password
from app.middleware.authentication import CurrentUserDep
from app.modules.iam.schema import LoginRequest, RefreshRequest, TokenOut, UserInfoOut
from app.modules.iam.service import IamService

router = APIRouter(prefix="/iam", tags=["IAM"])

DbDep = Annotated[AsyncSession, Depends(get_db)]
RedisDep = Annotated[Redis, Depends(get_redis)]


# 依赖注入工厂：创建 IamService 实例（绑定数据库会话与 Redis）
def get_service(db: DbDep, redis: RedisDep) -> IamService:
    return IamService(db, redis)


# 获取登录加密公钥：前端登录前先拉取，用公钥加密密码后提交
@router.get("/auth/public-key", response_model=ApiResponse[str], summary="获取登录加密公钥")
async def public_key() -> ApiResponse[str]:
    return success(data=load_public_key_pem())


# 登录接口：解密密码后校验用户名密码并签发令牌对
@router.post("/auth/login", response_model=ApiResponse[TokenOut], summary="登录")
async def login(
    req: LoginRequest,
    request: Request,
    service: Annotated[IamService, Depends(get_service)],
) -> ApiResponse[TokenOut]:
    # 前端用公钥加密密码传输，此处先用私钥解密还原明文，再交给 Service 走 bcrypt 校验
    req.password = rsa_decrypt_password(req.password)
    client_ip = request.client.host if request.client else None
    return success(data=await service.login(req, client_ip), message="登录成功")


# 刷新令牌接口：校验旧 refresh 并轮换签发新令牌对
@router.post("/auth/refresh", response_model=ApiResponse[TokenOut], summary="刷新令牌")
async def refresh(
    req: RefreshRequest,
    service: Annotated[IamService, Depends(get_service)],
) -> ApiResponse[TokenOut]:
    return success(data=await service.refresh(req), message="刷新成功")


# 登出接口：撤销 refresh 并拉黑当前 access（需携带有效 access 令牌）
@router.post("/auth/logout", response_model=ApiResponse[None], summary="登出")
async def logout(
    req: RefreshRequest,
    user: CurrentUserDep,
    service: Annotated[IamService, Depends(get_service)],
) -> ApiResponse[None]:
    await service.logout(user, req)
    return success(message="已退出登录")


# 当前用户信息接口：需携带有效的 access 令牌
@router.get("/auth/me", response_model=ApiResponse[UserInfoOut], summary="当前用户信息")
async def me(
    user: CurrentUserDep,
    service: Annotated[IamService, Depends(get_service)],
) -> ApiResponse[UserInfoOut]:
    if user.user_id is None:
        raise UnauthorizedError("未认证")
    return success(data=await service.me(user.user_id))
