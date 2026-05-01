"""Tests for /api/llm/discover-models — connectivity probe to LMStudio etc."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from neme_extractor.server.app import create_app


@dataclass
class _FakeResponse:
    status_code: int
    _payload: dict | None = None
    text: str = ""

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


@pytest.fixture
def app(tmp_path: Path):
    return create_app(state_dir=tmp_path / "state")


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_discover_models_returns_sorted_list(client, monkeypatch):
    def _fake_get(url, timeout=None, headers=None):
        assert url == "http://localhost:1234/v1/models"
        # No api_key in the request -> no Authorization header attached.
        assert not (headers or {}).get("Authorization")
        return _FakeResponse(
            status_code=200,
            _payload={"data": [
                {"id": "qwen2-vl-7b"},
                {"id": "llava-1.6-mistral"},
            ]},
        )

    monkeypatch.setattr("neme_extractor.llm.httpx.get", _fake_get)
    resp = await client.post(
        "/api/llm/discover-models",
        json={"endpoint": "http://localhost:1234"},
    )
    assert resp.status_code == 200
    assert resp.json()["models"] == ["llava-1.6-mistral", "qwen2-vl-7b"]


async def test_discover_models_forwards_api_key(client, monkeypatch):
    seen: dict = {}

    def _fake_get(url, timeout=None, headers=None):
        seen["headers"] = headers or {}
        return _FakeResponse(status_code=200, _payload={"data": []})

    monkeypatch.setattr("neme_extractor.llm.httpx.get", _fake_get)
    resp = await client.post(
        "/api/llm/discover-models",
        json={"endpoint": "https://api.openai.com", "api_key": "sk-test"},
    )
    assert resp.status_code == 200
    assert seen["headers"].get("Authorization") == "Bearer sk-test"


async def test_discover_models_422_on_unreachable(client, monkeypatch):
    def _fake_get(url, timeout=None, headers=None):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("neme_extractor.llm.httpx.get", _fake_get)
    resp = await client.post(
        "/api/llm/discover-models",
        json={"endpoint": "http://nope:9999"},
    )
    assert resp.status_code == 422
    assert "could not reach" in resp.json()["detail"]


async def test_discover_models_422_on_blank_endpoint(client):
    resp = await client.post(
        "/api/llm/discover-models", json={"endpoint": "   "},
    )
    assert resp.status_code == 422
