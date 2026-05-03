"""Tests for storage.character_copy.copy_character_to_project."""

from __future__ import annotations

from pathlib import Path

import pytest

from neme_anima.storage.character_copy import (
    CopyReport, copy_character_to_project,
)
from neme_anima.storage.metadata import FrameRecord, MetadataLog
from neme_anima.storage.project import Character, Project


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


def _seed_character(
    project: Project, *, name: str, slug: str | None = None,
    trigger: str = "", videos: list[str] | None = None,
    refs: dict[str, bytes] | None = None,
    frames: list[tuple[str, str, str | None]] | None = None,
) -> Character:
    """Helper: add a character + the artifacts attached to it.

    ``frames`` is a list of (filename_stem, video_stem, sidecar_text). Each
    frame writes the PNG, the optional sidecar, and a metadata row.
    """
    from neme_anima.storage.project import Character as _C  # type: ignore

    if slug is None:
        slug = name.lower().replace(" ", "-")
    if project.character_by_slug(slug):
        c = project.character_by_slug(slug)
    else:
        c = project.add_character(name=name, slug=slug)
    c.trigger_token = trigger
    project.save()

    for v in videos or []:
        # Use a real-ish path under tmp; make sure the file exists.
        vp = project.root.parent / f"{v}.mp4"
        vp.write_bytes(b"\x00")  # dummy
        try:
            project.add_source(vp)
        except ValueError:
            pass  # already added

    for filename, data in (refs or {}).items():
        project.add_ref_bytes(filename, data, character_slug=c.slug)

    log = MetadataLog(project.metadata_path)
    for stem, video_stem, sidecar in frames or []:
        (project.kept_dir / f"{stem}.png").write_bytes(b"\x89PNG")
        if sidecar is not None:
            (project.kept_dir / f"{stem}.txt").write_text(sidecar, encoding="utf-8")
        log.append(FrameRecord(
            filename=stem, kept=True, scene_idx=0, tracklet_id=0, frame_idx=0,
            timestamp_seconds=0.0, bbox=(0, 0, 1, 1), ccip_distance=0.0,
            sharpness=0.0, visibility=0.0, aspect=1.0, score=0.0,
            video_stem=video_stem, character_slug=c.slug,
        ))
    return c


def test_clean_copy_moves_everything(tmp_path):
    """No collisions. Every related artifact lands in dst, identity copied,
    counts match."""
    src = _make_project(tmp_path / "src", "src")
    _seed_character(
        src, name="Sora", slug="sora", trigger="sora_trig",
        videos=["ep01"],
        refs={"sora.png": b"\x89PNG-sora"},
        frames=[("ep01__a", "ep01", "tag1\n"), ("ep01__b", "ep01", "tag2\n")],
    )
    # Set core_tags + multiply on the source character to verify travel.
    src_char = src.character_by_slug("sora")
    src_char.core_tags = ["red eyes"]
    src_char.core_tags_enabled = True
    src_char.core_tags_freq_threshold = 0.42
    src_char.multiply = 1.5
    src.save()

    dst = _make_project(tmp_path / "dst", "dst")
    report = copy_character_to_project(
        src=src, src_character_slug="sora", dst=dst,
    )

    dst = Project.load(dst.root)  # reload for fresh state
    new_char = dst.character_by_slug("sora")
    assert new_char is not None
    assert new_char.name == "Sora"
    assert new_char.trigger_token == "sora_trig"
    assert new_char.core_tags == ["red eyes"]
    assert new_char.core_tags_enabled is True
    assert new_char.core_tags_freq_threshold == 0.42
    assert new_char.multiply == 1.5

    # Refs.
    assert len(new_char.refs) == 1
    ref_path = Path(new_char.refs[0].path)
    assert ref_path.is_file()
    assert ref_path.read_bytes() == b"\x89PNG-sora"

    # Sources.
    assert len(dst.sources) == 1
    assert Path(dst.sources[0].path).stem == "ep01"
    assert "ep01.mp4" in str(dst.sources[0].path)

    # Frames.
    assert (dst.kept_dir / "ep01__a.png").is_file()
    assert (dst.kept_dir / "ep01__b.png").is_file()
    assert (dst.kept_dir / "ep01__a.txt").read_text() == "tag1\n"

    # Metadata.
    rows = list(MetadataLog(dst.metadata_path).iter_records(
        character_slug="sora",
    ))
    assert len(rows) == 2

    # Report.
    assert report.character_slug == "sora"
    assert len(report.sources_added) == 1
    assert report.sources_skipped == []
    assert len(report.refs_added) == 1
    assert report.refs_renamed == {}
    assert len(report.frames_added) == 2
    assert report.frames_skipped == []
    assert report.metadata_rows_appended == 2
    assert report.dry_run is False


def test_source_video_collision_skips_source_keeps_frames(tmp_path):
    """When dst already has the source video at the same abs path, the
    source record is not duplicated. Frames are still imported (their
    individual filename collision rule applies separately)."""
    # Both projects reference the same video file on disk.
    shared_video = tmp_path / "ep01.mp4"
    shared_video.write_bytes(b"\x00")

    src = _make_project(tmp_path / "src", "src")
    src.add_source(shared_video)
    src_log = MetadataLog(src.metadata_path)
    (src.kept_dir / "ep01__a.png").write_bytes(b"\x89PNG")
    src_log.append(FrameRecord(
        filename="ep01__a", kept=True, scene_idx=0, tracklet_id=0, frame_idx=0,
        timestamp_seconds=0.0, bbox=(0, 0, 1, 1), ccip_distance=0.0,
        sharpness=0.0, visibility=0.0, aspect=1.0, score=0.0,
        video_stem="ep01", character_slug="default",
    ))
    src.add_character(name="Sora", slug="sora")
    # Move the frame to "sora" so the metadata reflects ownership.
    src_log.append(FrameRecord(
        filename="ep01__a", kept=True, scene_idx=0, tracklet_id=0, frame_idx=0,
        timestamp_seconds=0.0, bbox=(0, 0, 1, 1), ccip_distance=0.0,
        sharpness=0.0, visibility=0.0, aspect=1.0, score=0.0,
        video_stem="ep01", character_slug="sora",
    ))

    dst = _make_project(tmp_path / "dst", "dst")
    dst.add_source(shared_video)  # same abs path → collision

    report = copy_character_to_project(
        src=src, src_character_slug="sora", dst=dst,
    )

    dst = Project.load(dst.root)
    # Only one source record in dst.
    assert len(dst.sources) == 1
    assert report.sources_skipped == [str(shared_video.resolve())]
    assert report.sources_added == []
    # Frame was still copied (no per-frame collision).
    assert (dst.kept_dir / "ep01__a.png").is_file()
    assert "ep01__a" in report.frames_added
