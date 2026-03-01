import pytest
from httpx import AsyncClient, ASGITransport
import asyncio
from app.main import app
from typing import AsyncGenerator

@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac

@pytest.mark.asyncio
async def test_read_root(client: AsyncClient):
    response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "degraded"]
    assert "db" in data

@pytest.mark.asyncio
async def test_index_status(client: AsyncClient):
    response = await client.get("/index/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "progress_percent" in data
    assert data["progress_percent"] >= 0

@pytest.mark.asyncio
async def test_query_validation_error(client: AsyncClient):
    response = await client.post("/query", json={})
    assert response.status_code == 422
    assert "Validation error" in response.json().get("error", "")

@pytest.mark.asyncio
async def test_index_start_validation_error(client: AsyncClient):
    response = await client.post("/index/start", json={"folders": []})
    assert response.status_code == 400
    assert "No valid folder paths provided." in response.json().get("error", "")
