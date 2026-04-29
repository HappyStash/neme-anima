"""Tests for MetadataLog (append-only jsonl)."""

from __future__ import annotations

import json
from pathlib import Path

from neme_extractor.storage.metadata import MetadataLog, FrameRecord


def _rec(filename: str = "x.png", kept: bool = True, score: float = 0.5) -> FrameRecord:
    return FrameRecord(
        filename=filename, kept=kept,
        scene_idx=0, tracklet_id=1, frame_idx=42,
        timestamp_seconds=1.75,
        bbox=(10, 20, 100, 200),
        ccip_distance=0.1, sharpness=10.0, visibility=1.0, aspect=0.95,
        score=score,
        video_stem="ep01",
    )


def test_append_creates_file(tmp_path: Path):
    log = MetadataLog(tmp_path / "metadata.jsonl")
    log.append(_rec())
    assert (tmp_path / "metadata.jsonl").exists()


def test_append_writes_one_record_per_line(tmp_path: Path):
    log = MetadataLog(tmp_path / "metadata.jsonl")
    log.append(_rec("a.png"))
    log.append(_rec("b.png"))
    log.append(_rec("c.png"))
    lines = (tmp_path / "metadata.jsonl").read_text().strip().split("\n")
    assert len(lines) == 3
    decoded = [json.loads(line) for line in lines]
    assert [d["filename"] for d in decoded] == ["a.png", "b.png", "c.png"]


def test_iter_records_streams_all(tmp_path: Path):
    log = MetadataLog(tmp_path / "metadata.jsonl")
    log.append(_rec("a.png"))
    log.append(_rec("b.png", kept=False))
    records = list(log.iter_records())
    assert len(records) == 2
    assert records[0].filename == "a.png" and records[0].kept is True
    assert records[1].filename == "b.png" and records[1].kept is False


def test_iter_records_handles_missing_file(tmp_path: Path):
    log = MetadataLog(tmp_path / "metadata.jsonl")
    assert list(log.iter_records()) == []


def test_filter_by_video_stem(tmp_path: Path):
    log = MetadataLog(tmp_path / "metadata.jsonl")
    log.append(_rec("a.png"))
    rec_other = _rec("b.png")
    rec_other.video_stem = "ep02"
    log.append(rec_other)
    only_ep01 = list(log.iter_records(video_stem="ep01"))
    assert len(only_ep01) == 1 and only_ep01[0].filename == "a.png"
