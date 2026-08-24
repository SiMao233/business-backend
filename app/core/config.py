"""应用全局配置。

所有配置项均支持通过环境变量或 `.env` 文件覆盖（pydantic-settings v2）。
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置模型。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- 应用 ----
    app_name: str = "Business Backend"
    app_env: Literal["development", "testing", "production"] = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_api_prefix: str = "/api/v1"
    # 注意：该字段仅用于 docker-compose 的容器系统时区（TZ），Python 代码不读取它。
    # 应用内时间基准统一为 UTC（存储/输出均 UTC），前端负责转换为本地时区展示。
    app_timezone: str = "Asia/Shanghai"

    # 初始超管账号密码（仅 seed 初始化首次创建时使用，生产务必通过 .env 覆盖为强密码）
    app_admin_password: str = "Admin@123456"

    # ---- 数据库 (MySQL 8) ----
    # 注意：以下仅为本地开发占位默认值，不含任何真实凭据。
    # 真实连接信息请通过 .env 注入（.env 已被 .gitignore 忽略，不会提交到 Git）。
    db_host: str = "127.0.0.1"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str = ""
    db_name: str = "business_backend"
    db_echo: bool = False
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_recycle: int = 3600

    # ---- Redis ----
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None

    # ---- JWT / 安全 ----
    # 生产环境务必在 .env 中设置强随机密钥，切勿使用默认值
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # ---- 登录密码 RSA 加密 ----
    # 密钥对通过 `openssl genrsa` 生成，位于项目根 keys/ 目录（已 gitignore，勿提交）
    # 前端使用公钥加密密码，后端使用私钥解密后走 bcrypt 校验
    rsa_private_key_path: str = "keys/private.pem"
    rsa_public_key_path: str = "keys/public.pem"

    # ---- CORS ----
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # ---- 文件存储 ----
    # 本地存储根目录（相对项目根；生产可改为对象存储，见 app/modules/file/storage.py 抽象）
    file_storage_dir: str = "storage"
    # 单文件大小上限（字节），默认 10MB
    file_max_size: int = 10 * 1024 * 1024
    # 允许上传的扩展名白名单（不含点）
    file_allowed_extensions: list[str] = Field(
        default_factory=lambda: [
            "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
            "txt", "md", "csv",
            "png", "jpg", "jpeg", "gif", "webp", "svg",
            "mp3", "mp4", "zip",
        ]
    )
    # 可选：对外公开访问的基础 URL（如 CDN / 网关前缀），为空则返回相对下载路径
    file_public_base_url: str = ""

    # ---- 日志 ----
    log_level: str = "INFO"

    @property
    def database_url(self) -> str:
        """构造 MySQL 异步连接串（aiomysql 驱动）。"""
        password = self.db_password or ""
        return (
            f"mysql+aiomysql://{self.db_user}:{password}@{self.db_host}:{self.db_port}"
            f"/{self.db_name}?charset=utf8mb4"
        )

    @property
    def redis_url(self) -> str:
        """构造 Redis 连接串。"""
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    """获取全局配置单例（进程内缓存，避免重复解析 .env）。"""
    return Settings()
