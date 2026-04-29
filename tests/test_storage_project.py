"""Tests for the Project storage class."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from neme_extractor.storage.project import Project, Source, RefImage


def test_create_initializes_folder_structure(tmp_path: Path):
    p = Project.create(tmp_path / "megumin", name="megumin")
    assert (tmp_path / "megumin" / "project.json").exists()
    assert (tmp_path / "megumin" / "refs").is_dir()
    assert (tmp_path / "megumin" / "output" / "kept").is_dir()
    assert (tmp_path / "megumin" / "output" / "rejected").is_dir()
    assert (tmp_path / "megumin" / "output" / "cache").is_dir()
    assert p.name == "megumin"
    assert p.slug == "megumin"
    assert p.sources == []
    assert p.refs == []


def test_load_roundtrips(tmp_path: Path):
    Project.create(tmp_path / "p", name="p")
    p = Project.load(tmp_path / "p")
    assert p.name == "p"
    assert p.slug == "p"
    assert isinstance(p.created_at, datetime)


def test_create_rejects_existing_folder(tmp_path: Path):
    (tmp_path / "exists").mkdir()
    with pytest.raises(FileExistsError):
        Project.create(tmp_path / "exists", name="x")


def test_slug_is_filesystem_safe(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="Megumin's WIP / draft")
    # Slug should be the folder name, not the display name.
    assert p.slug == "p"
    assert p.name == "Megumin's WIP / draft"


def test_save_roundtrips_thresholds_overrides(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="p")
    p.thresholds_overrides = {"identify": {"body_max_distance_loose": 0.22}}
    p.save()
    reloaded = Project.load(p.root)
    assert reloaded.thresholds_overrides == {"identify": {"body_max_distance_loose": 0.22}}
