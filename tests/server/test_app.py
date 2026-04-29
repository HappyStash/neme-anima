"""Tests for the FastAPI app factory + lifespan."""

from __future__ import annotations

from pathlib import Path

from httpx import ASGITransport, AsyncClient

from neme_extractor.server.app import create_app


async def test_app_starts_and_serves_health(tmp_path: Path):
    app = create_app(state_dir=tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}


async def test_app_exposes_registry_and_queue_in_state(tmp_path: Path):
    app = create_app(state_dir=tmp_path)
    assert app.state.registry is not None
    assert app.state.queue is not None
    assert app.state.broadcaster is not None
