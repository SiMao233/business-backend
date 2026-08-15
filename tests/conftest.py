"""pytest 全局夹具。"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def client() -> AsyncClient:
    """提供基于 ASGI 的异步测试客户端（ASGITransport 不触发 lifespan）。"""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
