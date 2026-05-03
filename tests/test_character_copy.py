"""Tests for storage.character_copy.copy_character_to_project."""

from __future__ import annotations

from pathlib import Path

import pytest

from neme_anima.storage.character_copy import (
    CopyReport, copy_character_to_project,
)
from neme_anima.storage.project import Project


def _make_project(root: Path, name: str) -> Project:
    return Project.create(root, name=name)


def test_copy_refuses_when_dst_already_has_slug(tmp_path):
    src = _make_project(tmp_path / "src", "src")
    dst = _make_project(tmp_path / "dst", "dst")
    # Both projects start with a "default" character — guaranteed slug collision.
    with pytest.raises(ValueError, match="already exists"):
        copy_character_to_project(
            src=src, src_character_slug="default", dst=dst,
        )
    # No partial state in dst.
    dst_reloaded = Project.load(dst.root)
    assert len(dst_reloaded.characters) == 1


def test_copy_raises_on_unknown_source_character(tmp_path):
    src = _make_project(tmp_path / "src", "src")
    dst = _make_project(tmp_path / "dst", "dst")
    with pytest.raises(KeyError, match="ghost"):
        copy_character_to_project(
            src=src, src_character_slug="ghost", dst=dst,
        )
