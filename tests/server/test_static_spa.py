"""Verify the FastAPI app serves the SPA at /."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from neme_anima.server.app import create_app


async def test_root_serves_spa_index(tmp_path: Path):
    # Construct a minimal fake static dir with an index.html so the test
    # works even before the real frontend is built.
    static_dir = Path(__file__).parent.parent.parent / "src" / "neme_anima" / "server" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    index = static_dir / "index.html"
    if not index.exists():
        index.write_text("<!doctype html><title>test</title>", encoding="utf-8")
    try:
        app = create_app(state_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            assert "<!doctype html>" in resp.text.lower() or "<html" in resp.text.lower()
    finally:
        # Clean up only if we created it as the placeholder.
        if index.read_text() == "<!doctype html><title>test</title>":
            index.unlink()


async def test_unknown_spa_route_falls_back_to_index(tmp_path: Path):
    static_dir = Path(__file__).parent.parent.parent / "src" / "neme_anima" / "server" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    index = static_dir / "index.html"
    placeholder = False
    if not index.exists():
        index.write_text("<!doctype html><title>fallback</title>", encoding="utf-8")
        placeholder = True
    try:
        app = create_app(state_dir=tmp_path)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/some/spa/route")
            assert resp.status_code == 200
            assert "<!doctype html>" in resp.text.lower() or "<html" in resp.text.lower()
    finally:
        if placeholder:
            index.unlink()


async def test_unknown_api_route_does_not_fall_back(tmp_path: Path):
    """An unknown /api/* path must 404, not return the SPA."""
    app = create_app(state_dir=tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/does-not-exist")
        assert resp.status_code == 404
