"""Tests for /api/queue routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from neme_extractor.server.app import create_app


@pytest.fixture
def app(tmp_path: Path):
    return create_app(state_dir=tmp_path / "state")


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        async with app.router.lifespan_context(app):
            yield c


async def test_list_queue_empty(client):
    resp = await client.get("/api/queue")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_cancel_unknown_returns_404(client):
    resp = await client.delete("/api/queue/does-not-exist")
    assert resp.status_code == 404
