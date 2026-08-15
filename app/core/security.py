"""安全工具：JWT 令牌签发/校验与密码哈希。

提供访问令牌（access）与刷新令牌（refresh）的签发/解析，以及 bcrypt 密码哈希。
认证业务（登录、刷新、登出、当前用户）由 IAM 模块实现。
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import bcrypt
import jwt

from app.core.config import get_settings

settings = get_settings()


def create_access_token(
    subject: str, expires_minutes: int | None = None, version: int | None = None
) -> str:
    """签发 JWT 访问令牌。`subject` 一般为用户标识；`version` 为令牌版本号（强制下线用）。"""
    now = datetime.now(UTC)
    expire = now + timedelta(minutes=expires_minutes or settings.jwt_access_token_expire_minutes)
    payload = {
        "sub": subject,
        "token_type": "access",
        "jti": uuid4().hex,
        "iat": now,
        "exp": expire,
    }
    if version is not None:
        payload["ver"] = version
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(
    subject: str, expires_days: int | None = None, version: int | None = None
) -> str:
    """签发 JWT 刷新令牌（带 `token_type=refresh` 与 `jti` 声明，供 Redis 有状态管理）。

    `version` 为令牌版本号，用于强制下线：版本号一旦变化，该用户所有已发令牌立即失效。
    """
    now = datetime.now(UTC)
    expire = now + timedelta(days=expires_days or settings.jwt_refresh_token_expire_days)
    payload = {
        "sub": subject,
        "token_type": "refresh",
        "jti": uuid4().hex,
        "iat": now,
        "exp": expire,
    }
    if version is not None:
        payload["ver"] = version
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict[str, object]:
    """解析并校验 JWT，返回 payload。

    校验失败（过期/签名错误等）抛出 `jwt.PyJWTError` 子类。
    """
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])


def hash_password(plain_password: str) -> str:
    """使用 bcrypt 生成密码哈希（自动加密）。"""
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """校验明文密码与哈希是否匹配。"""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
