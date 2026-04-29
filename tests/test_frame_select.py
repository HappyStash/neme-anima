"""Smoke test for frame_select.py — pick best frames from a synthetic tracklet."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from imgutils.metrics import ccip_extract_feature

from neme_extractor.config import FrameSelectConfig
from neme_extractor.detect import Detection, DetectionKind
from neme_extractor.frame_select import select_frames
from neme_extractor.track import TrackedDetection, Tracklet
from neme_extractor.video import Video


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    """A 60-frame clip at 24 fps with varied content per frame."""
    p = tmp_path / "clip.mp4"
    h, w, fps = 480, 640, 24
    writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(42)
    for i in range(60):
        f = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        # vary brightness across frames
        f = (f.astype(int) // 2 + i * 4).clip(0, 255).astype(np.uint8)
        writer.write(f)
    writer.release()
    return p


def test_select_frames_short_tracklet_picks_one(clip: Path):
    video = Video(clip)
    items = []
    for fi in range(0, 12):  # 0.5 s tracklet
        items.append(TrackedDetection(
            scene_idx=0, tracklet_id=1, frame_idx=fi,
            detection=Detection(DetectionKind.PERSON, 100, 80, 300, 400, "person", 0.9),
        ))
    tracklet = Tracklet(scene_idx=0, tracklet_id=1, items=tuple(items))
    rng = np.random.default_rng(7)
    ref = ccip_extract_feature(Image.fromarray(
        rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    ))

    cfg = FrameSelectConfig(short_tracklet_seconds=1.0, long_tracklet_seconds=5.0,
                            top_k_short=1, top_k_long=3, dedup_min_frame_gap=4)
    picks = select_frames(tracklet, video, [ref], cfg)
    assert len(picks) == 1
    assert picks[0].scene_idx == 0
    assert picks[0].tracklet_id == 1
    assert 0 <= picks[0].frame_idx < 12


def test_select_frames_long_tracklet_picks_more(clip: Path):
    video = Video(clip)
    items = []
    for fi in range(0, 60):  # 2.5 s tracklet
        items.append(TrackedDetection(
            scene_idx=0, tracklet_id=1, frame_idx=fi,
            detection=Detection(DetectionKind.PERSON, 100, 80, 300, 400, "person", 0.9),
        ))
    tracklet = Tracklet(scene_idx=0, tracklet_id=1, items=tuple(items))
    rng = np.random.default_rng(7)
    ref = ccip_extract_feature(Image.fromarray(
        rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    ))

    cfg = FrameSelectConfig(short_tracklet_seconds=1.0, long_tracklet_seconds=5.0,
                            top_k_short=1, top_k_long=3, dedup_min_frame_gap=4)
    picks = select_frames(tracklet, video, [ref], cfg)
    # 2.5 s is between short and long; expect interpolated value in [1, 3].
    assert 1 <= len(picks) <= 3
    # Picks must be in distinct frames.
    assert len({p.frame_idx for p in picks}) == len(picks)
