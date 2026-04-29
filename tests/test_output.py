"""Tests for OutputWriter (project-centric)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neme_extractor.output import OutputWriter, ProjectFrameRecord
from neme_extractor.storage.metadata import FrameRecord, MetadataLog
from neme_extractor.storage.project import Project


def _make_project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "p", name="p")


def _record(filename: str, kept: bool, video_stem: str) -> FrameRecord:
    return FrameRecord(
        filename=filename, kept=kept,
        scene_idx=0, tracklet_id=1, frame_idx=42,
        timestamp_seconds=1.75,
        bbox=(0, 0, 32, 32),
        ccip_distance=0.1, sharpness=10.0, visibility=1.0, aspect=0.95,
        score=0.8,
        video_stem=video_stem,
    )


def test_filename_includes_video_stem_prefix():
    name = OutputWriter.filename_for(video_stem="ep01", scene_idx=3, tracklet_id=12, frame_idx=847)
    assert name == "ep01__s003_t012_f000847"


def test_kept_writes_into_unified_kept_folder(tmp_path: Path):
    project = _make_project(tmp_path)
    writer = OutputWriter(project=project, video_stem="ep01")
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    rec = _record("ep01__s000_t001_f000010", kept=True, video_stem="ep01")
    p = writer.write_kept(rec, img, "tag1, tag2")
    assert p == project.kept_dir / "ep01__s000_t001_f000010.png"
    assert p.exists()
    assert p.with_suffix(".txt").read_text() == "tag1, tag2\n"


def test_rejected_writes_no_txt(tmp_path: Path):
    project = _make_project(tmp_path)
    writer = OutputWriter(project=project, video_stem="ep01")
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    rec = _record("ep01__s000_t002_f000024", kept=False, video_stem="ep01")
    p = writer.write_rejected(rec, img)
    assert p == project.rejected_dir / "ep01__s000_t002_f000024.png"
    assert p.exists()
    assert not p.with_suffix(".txt").exists()


def test_metadata_appends_to_jsonl(tmp_path: Path):
    project = _make_project(tmp_path)
    writer = OutputWriter(project=project, video_stem="ep01")
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    rec = _record("ep01__s000_t001_f000010", kept=True, video_stem="ep01")
    writer.write_kept(rec, img, "")
    log = MetadataLog(project.metadata_path)
    rows = list(log.iter_records())
    assert len(rows) == 1 and rows[0].filename == "ep01__s000_t001_f000010"


def test_multiple_videos_unify_into_same_kept_folder(tmp_path: Path):
    project = _make_project(tmp_path)
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    w1 = OutputWriter(project=project, video_stem="ep01")
    w1.write_kept(_record("ep01__s000_t001_f000010", True, "ep01"), img, "a")
    w2 = OutputWriter(project=project, video_stem="ep02")
    w2.write_kept(_record("ep02__s000_t001_f000005", True, "ep02"), img, "b")
    files = sorted(p.name for p in project.kept_dir.glob("*.png"))
    assert files == [
        "ep01__s000_t001_f000010.png",
        "ep02__s000_t001_f000005.png",
    ]
