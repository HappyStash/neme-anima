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


def test_add_source_appends_with_timestamp(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="p")
    fake_vid = tmp_path / "ep01.mkv"
    fake_vid.write_bytes(b"")
    p.add_source(fake_vid)
    assert len(p.sources) == 1
    assert Path(p.sources[0].path) == fake_vid.resolve()
    assert p.sources[0].excluded_refs == []
    assert p.sources[0].added_at  # set


def test_add_source_rejects_duplicates(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="p")
    fake_vid = tmp_path / "ep01.mkv"
    fake_vid.write_bytes(b"")
    p.add_source(fake_vid)
    with pytest.raises(ValueError, match="already in project"):
        p.add_source(fake_vid)


def test_add_ref_appends(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="p")
    img = tmp_path / "ref.png"
    img.write_bytes(b"")
    p.add_ref(img)
    assert len(p.refs) == 1
    assert Path(p.refs[0].path) == img.resolve()


def test_remove_ref_strips_from_excluded_lists(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="p")
    img1 = tmp_path / "ref1.png"; img1.write_bytes(b"")
    img2 = tmp_path / "ref2.png"; img2.write_bytes(b"")
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    p.add_ref(img1)
    p.add_ref(img2)
    p.add_source(vid)
    p.set_excluded_refs(0, [str(img2.resolve())])
    assert p.sources[0].excluded_refs == [str(img2.resolve())]
    p.remove_ref(str(img2.resolve()))
    # img2 removed from project AND from any excluded_refs list.
    assert len(p.refs) == 1
    assert p.sources[0].excluded_refs == []


def test_set_excluded_refs_persists(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="p")
    img = tmp_path / "ref.png"; img.write_bytes(b"")
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    p.add_ref(img)
    p.add_source(vid)
    p.set_excluded_refs(0, [str(img.resolve())])
    p.save()
    reloaded = Project.load(p.root)
    assert reloaded.sources[0].excluded_refs == [str(img.resolve())]
