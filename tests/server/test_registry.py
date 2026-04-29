"""Tests for the SQLite project registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from neme_extractor.server.registry import ProjectRegistry, RegistryEntry
from neme_extractor.storage.project import Project


@pytest.fixture
def registry(tmp_path: Path) -> ProjectRegistry:
    return ProjectRegistry(tmp_path / "db.sqlite")


def test_empty_registry_lists_nothing(registry: ProjectRegistry):
    assert registry.list() == []


def test_register_appends_entry(registry: ProjectRegistry, tmp_path: Path):
    proj = Project.create(tmp_path / "p", name="p")
    registry.register(proj)
    rows = registry.list()
    assert len(rows) == 1
    assert rows[0].slug == "p"
    assert rows[0].name == "p"
    assert Path(rows[0].folder) == proj.root.resolve()


def test_register_idempotent_on_same_slug(registry: ProjectRegistry, tmp_path: Path):
    proj = Project.create(tmp_path / "p", name="p")
    registry.register(proj)
    # Re-registering refreshes name + last_opened, but doesn't duplicate.
    registry.register(proj)
    assert len(registry.list()) == 1


def test_get_returns_entry_by_slug(registry: ProjectRegistry, tmp_path: Path):
    proj = Project.create(tmp_path / "p", name="display name")
    registry.register(proj)
    got = registry.get("p")
    assert got is not None
    assert got.name == "display name"


def test_get_missing_returns_none(registry: ProjectRegistry):
    assert registry.get("nope") is None


def test_unregister_removes_entry(registry: ProjectRegistry, tmp_path: Path):
    proj = Project.create(tmp_path / "p", name="p")
    registry.register(proj)
    registry.unregister("p")
    assert registry.list() == []


def test_touch_updates_last_opened(registry: ProjectRegistry, tmp_path: Path):
    proj = Project.create(tmp_path / "p", name="p")
    registry.register(proj)
    before = registry.get("p").last_opened_at
    registry.touch("p")
    after = registry.get("p").last_opened_at
    assert after >= before


def test_list_orders_by_last_opened_desc(registry: ProjectRegistry, tmp_path: Path):
    a = Project.create(tmp_path / "a", name="a"); registry.register(a)
    b = Project.create(tmp_path / "b", name="b"); registry.register(b)
    registry.touch("a")  # a most recent
    rows = registry.list()
    assert [r.slug for r in rows] == ["a", "b"]
