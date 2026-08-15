"""健康检查与系统接口基础测试。

说明：就绪检查（/api/v1/system/ready）在 MySQL / Redis 未启动时会返回 `degraded`，
因此本测试仅断言统一响应结构与检查项集合，不依赖外部基础设施可用性。
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_root_health(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["data"]["status"] == "ok"


@pytest.mark.asyncio
async def test_system_health(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/system/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["data"]["service"] == "business-backend"


@pytest.mark.asyncio
async def test_system_ready(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/system/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert set(body["data"]["checks"].keys()) == {"database", "redis"}
    assert body["data"]["status"] in {"ok", "degraded"}
