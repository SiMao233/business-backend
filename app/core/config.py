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
    # Redis 命令/连接超时（秒）：远程 Redis 不稳定时快速失败，避免请求无限挂起
    # 云 NAT/防火墙会静默丢弃空闲连接，超时需给网络抖动留余量，故设为 5s
    redis_socket_timeout: float = 5.0
    redis_socket_connect_timeout: float = 5.0

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

    # ---- AI 能力（内嵌模块 app/modules/ai/）----
    # 注意：真实模型 API Key 由模型供应商（sys_model_provider.api_key）配置，
    # 此处仅放非敏感占位默认值；生产环境通过 .env 覆盖。
    # 默认模型标识（Agent 未绑定模型实例时的兜底；为空则要求 Agent 必须绑定）
    ai_default_model: str = ""
    # 单次 LLM 调用超时（秒）
    ai_request_timeout: float = 60.0
    # 失败重试次数
    ai_max_retries: int = 2
    # 对话最大历史轮数（保留最近 N 轮，用于上下文组装）
    ai_max_history_rounds: int = 10
    # 流式对话是否向模型请求 token 用量（OpenAI 兼容网关需支持 stream_options.include_usage；
    # 若网关不支持会返回 400，可在 .env 设 AI_STREAM_USAGE=false 关闭）
    ai_stream_usage: bool = True

    # ---- 联网搜索（web_search 工具）----
    # 搜索服务 API Key（Tavily / 阿里云百炼 / 博查等，.env 注入，勿硬编码）
    web_search_api_key: str = ""
    # 单次搜索返回结果条数
    web_search_max_results: int = 5

    # ---- 向量检索（Qdrant，RAG）----
    # Qdrant 服务地址（本机 docker 为 127.0.0.1:6333；远程/容器内按部署调整）
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    # 是否使用 HTTPS（远程 Qdrant Cloud 需 true）
    qdrant_https: bool = False
    # API Key（Qdrant Cloud / 自建鉴权时使用，本地可留空）
    qdrant_api_key: str | None = None
    # 单一 collection 名（所有知识库向量共存，payload 按 knowledge_base_id 过滤）
    qdrant_collection: str = "knowledge_chunks"

    # ---- 文档切分（RAG 索引）----
    # 切分块大小（字符）与重叠
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ---- RAG 检索 ----
    # 每次检索返回的 top-k 块数
    rag_top_k: int = 4
    # 相似度阈值（低于该分数的结果丢弃；0~1，越大越严格）
    # 注意：文本 embedding 的余弦相似度普遍偏低（相关片段通常 0.25~0.6，无关 0.1~0.25），
    # 设 0.7 会导致几乎永远 0 命中（表现为"没检索/检索不到"），故默认取 0.3，可按模型实测调整。
    rag_score_threshold: float = 0.3

    # ---- 用量统计 ----
    # 业务时区：用于把 UTC 时间切成"自然日"（usage_date）后按天聚合；
    # 仅用量统计模块读取，与 app_timezone（仅供容器 TZ）相互独立。
    usage_timezone: str = "Asia/Shanghai"

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
