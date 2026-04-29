"""Tests for output.py — cache round-trip and image+sidecar writes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neme_extractor.config import Thresholds
from neme_extractor.detect import Detection, DetectionKind
from neme_extractor.output import FrameRecord, OutputWriter
from neme_extractor.track import TrackedDetection, Tracklet
from neme_extractor.video import Scene


def _make_tracklet(scene_idx: int, tracklet_id: int, frames: list[int]) -> Tracklet:
    items = tuple(
        TrackedDetection(
            scene_idx=scene_idx,
            tracklet_id=tracklet_id,
            frame_idx=fi,
            detection=Detection(
                kind=DetectionKind.PERSON,
                x1=10, y1=20, x2=110, y2=220,
                label="person", score=0.9,
            ),
        )
        for fi in frames
    )
    return Tracklet(scene_idx=scene_idx, tracklet_id=tracklet_id, items=items)


def test_kept_image_writes_png_and_txt(tmp_path: Path):
    w = OutputWriter(root=tmp_path, video_stem="vid")
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    rec = FrameRecord(
        filename=OutputWriter.filename_for(0, 1, 12),
        kept=True, scene_idx=0, tracklet_id=1, frame_idx=12,
        timestamp_seconds=0.5,
        bbox=(0, 0, 32, 32),
        ccip_distance=0.12, sharpness=10.0, visibility=1.0, aspect=0.95, score=0.8,
    )
    p = w.write_kept(rec, img, "1girl, smile")
    assert p.exists()
    assert p.with_suffix(".txt").read_text(encoding="utf-8") == "1girl, smile\n"


def test_rejected_image_no_txt(tmp_path: Path):
    w = OutputWriter(root=tmp_path, video_stem="vid")
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    rec = FrameRecord(
        filename=OutputWriter.filename_for(0, 2, 24),
        kept=False, scene_idx=0, tracklet_id=2, frame_idx=24,
        timestamp_seconds=1.0,
        bbox=(0, 0, 32, 32),
        ccip_distance=0.50, sharpness=8.0, visibility=1.0, aspect=0.9, score=0.4,
    )
    p = w.write_rejected(rec, img)
    assert p.exists()
    assert not p.with_suffix(".txt").exists()


def test_thresholds_roundtrip(tmp_path: Path):
    w = OutputWriter(root=tmp_path, video_stem="vid")
    t = Thresholds()
    w.write_thresholds(t)
    t2 = w.read_thresholds()
    assert t2.detect.person_score_min == t.detect.person_score_min


def test_scenes_cache_roundtrip(tmp_path: Path):
    w = OutputWriter(root=tmp_path, video_stem="vid")
    scenes = [Scene(0, 0, 100), Scene(1, 100, 240), Scene(2, 240, 400)]
    w.write_scenes(scenes)
    out = w.read_scenes()
    assert out == scenes


def test_tracklets_cache_roundtrip(tmp_path: Path):
    w = OutputWriter(root=tmp_path, video_stem="vid")
    tracklets = [
        _make_tracklet(0, 1, [0, 1, 2, 3]),
        _make_tracklet(0, 2, [0, 1, 2]),
        _make_tracklet(1, 1, [10, 11, 12]),
    ]
    w.write_tracklets(tracklets)
    out = w.read_tracklets()
    assert len(out) == 3
    keys_in = {(t.scene_idx, t.tracklet_id) for t in tracklets}
    keys_out = {(t.scene_idx, t.tracklet_id) for t in out}
    assert keys_in == keys_out
    # Frame indices preserved per tracklet.
    by_key = {(t.scene_idx, t.tracklet_id): [it.frame_idx for it in t.items] for t in out}
    assert by_key[(0, 1)] == [0, 1, 2, 3]
    assert by_key[(1, 1)] == [10, 11, 12]


def test_metadata_flushes(tmp_path: Path):
    w = OutputWriter(root=tmp_path, video_stem="vid")
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    rec = FrameRecord(
        filename="s000_t001_f000010", kept=True,
        scene_idx=0, tracklet_id=1, frame_idx=10,
        timestamp_seconds=0.42,
        bbox=(0, 0, 10, 10),
        ccip_distance=0.1, sharpness=5.0, visibility=1.0, aspect=0.9, score=0.7,
    )
    w.write_kept(rec, img, "tag1")
    p = w.flush_metadata()
    import json
    data = json.loads(p.read_text())
    assert len(data) == 1
    assert data[0]["filename"] == "s000_t001_f000010"
    assert data[0]["kept"] is True


def test_run_header_roundtrip(tmp_path: Path):
    w = OutputWriter(root=tmp_path, video_stem="vid")
    fake_vid = tmp_path / "fake.mp4"
    fake_vid.write_bytes(b"")
    w.write_run_header(fake_vid, fps=24.0, num_frames=120, refs=[])
    h = w.read_run_header()
    assert h["fps"] == 24.0
    assert h["num_frames"] == 120
