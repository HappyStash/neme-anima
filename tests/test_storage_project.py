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


def test_add_ref_copies_into_project(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="p")
    img = tmp_path / "ref.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    r = p.add_ref(img)
    # The stored path is the project-internal copy, not the source path.
    assert Path(r.path).parent == (tmp_path / "p" / "refs").resolve()
    assert Path(r.path).read_bytes() == b"\x89PNG\r\n\x1a\n"
    # Original is untouched.
    assert img.exists()


def test_add_ref_uniquifies_collisions(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="p")
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    (a / "ref.png").write_bytes(b"AAAA")
    (b / "ref.png").write_bytes(b"BBBB")
    r1 = p.add_ref(a / "ref.png")
    r2 = p.add_ref(b / "ref.png")
    assert Path(r1.path).name == "ref.png"
    assert Path(r2.path).name == "ref-2.png"
    assert Path(r1.path).read_bytes() == b"AAAA"
    assert Path(r2.path).read_bytes() == b"BBBB"


def test_add_ref_bytes(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="p")
    r = p.add_ref_bytes("dropped.jpg", b"\xff\xd8\xff")
    assert Path(r.path).read_bytes() == b"\xff\xd8\xff"
    assert Path(r.path).parent == (tmp_path / "p" / "refs").resolve()


def test_remove_ref_strips_from_excluded_lists_and_deletes_file(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="p")
    img1 = tmp_path / "ref1.png"; img1.write_bytes(b"AAA")
    img2 = tmp_path / "ref2.png"; img2.write_bytes(b"BBB")
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    r1 = p.add_ref(img1)
    r2 = p.add_ref(img2)
    p.add_source(vid)
    p.set_excluded_refs(0, [r2.path])
    assert p.sources[0].excluded_refs == [r2.path]
    p.remove_ref(r2.path)
    assert len(p.refs) == 1
    assert p.sources[0].excluded_refs == []
    # File on disk is gone too.
    assert not Path(r2.path).exists()
    # The other ref is untouched.
    assert Path(r1.path).exists()


def test_set_excluded_refs_persists(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="p")
    img = tmp_path / "ref.png"; img.write_bytes(b"X")
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    r = p.add_ref(img)
    p.add_source(vid)
    p.set_excluded_refs(0, [r.path])
    p.save()
    reloaded = Project.load(p.root)
    assert reloaded.sources[0].excluded_refs == [r.path]


def test_effective_refs_excludes_per_video_opt_outs(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="p")
    a = tmp_path / "a.png"; a.write_bytes(b"A")
    b = tmp_path / "b.png"; b.write_bytes(b"B")
    c = tmp_path / "c.png"; c.write_bytes(b"C")
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    ra = p.add_ref(a); rb = p.add_ref(b); rc = p.add_ref(c)
    p.add_source(vid)
    p.set_excluded_refs(0, [rb.path])
    eff = p.effective_refs_for(0)
    assert sorted(eff) == sorted([ra.path, rc.path])


def test_effective_refs_default_is_all_project_refs(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="p")
    a = tmp_path / "a.png"; a.write_bytes(b"A")
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    ra = p.add_ref(a)
    p.add_source(vid)
    eff = p.effective_refs_for(0)
    assert eff == [ra.path]


def test_import_videos_from_folder(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="p")
    folder = tmp_path / "videos"; folder.mkdir()
    (folder / "ep01.mkv").write_bytes(b"")
    (folder / "ep02.MP4").write_bytes(b"")
    (folder / "notes.txt").write_text("ignore me")
    (folder / "subdir").mkdir()  # ignored — non-recursive
    added, skipped = p.import_videos_from_folder(folder)
    names = sorted(Path(s.path).name for s in added)
    assert names == ["ep01.mkv", "ep02.MP4"]
    assert skipped == []
    assert p.source_root == str(folder.resolve())
    # Re-running on the same folder skips existing entries instead of dup-adding.
    added2, skipped2 = p.import_videos_from_folder(folder)
    assert added2 == []
    assert len(skipped2) == 2


def test_import_videos_persists_source_root(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="p")
    folder = tmp_path / "videos"; folder.mkdir()
    (folder / "ep01.mkv").write_bytes(b"")
    p.import_videos_from_folder(folder)
    reloaded = Project.load(p.root)
    assert reloaded.source_root == str(folder.resolve())


def test_effective_refs_paths_helpers(tmp_path: Path):
    p = Project.create(tmp_path / "p", name="p")
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    p.add_source(vid)
    assert p.video_stem(0) == "ep01"
    assert p.kept_dir == p.root / "output" / "kept"
    assert p.rejected_dir == p.root / "output" / "rejected"
    assert p.cache_dir_for("ep01") == p.root / "output" / "cache" / "ep01"
    assert p.metadata_path == p.root / "output" / "metadata.jsonl"
