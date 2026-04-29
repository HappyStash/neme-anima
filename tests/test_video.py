"""Smoke test for video.py: synthesize a 2-scene clip, verify reader + scene detection."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from neme_extractor.video import Video, detect_scenes


@pytest.fixture
def two_scene_clip(tmp_path: Path) -> Path:
    """Generate a tiny 2-scene MP4: 30 blue frames then 30 red frames at 24 fps."""
    out = tmp_path / "two_scenes.mp4"
    h, w, fps = 240, 320, 24
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    assert writer.isOpened(), "OpenCV could not open mp4v writer"

    blue = np.zeros((h, w, 3), dtype=np.uint8)
    blue[:] = (200, 60, 30)  # BGR
    red = np.zeros((h, w, 3), dtype=np.uint8)
    red[:] = (30, 60, 200)  # BGR

    for i in range(30):
        f = blue.copy()
        # Moving white square so content isn't trivially constant.
        x = 20 + i * 4
        cv2.rectangle(f, (x, 80), (x + 40, 120), (255, 255, 255), -1)
        writer.write(f)
    for i in range(30):
        f = red.copy()
        x = 280 - i * 4
        cv2.rectangle(f, (x, 120), (x + 40, 160), (255, 255, 255), -1)
        writer.write(f)
    writer.release()
    assert out.exists() and out.stat().st_size > 0
    return out


def test_video_reads_metadata(two_scene_clip: Path):
    v = Video(two_scene_clip)
    assert len(v) == 60
    assert abs(v.fps - 24.0) < 0.1
    assert abs(v.duration_seconds - 2.5) < 0.5  # encoder may add a small fudge


def test_video_get_returns_rgb(two_scene_clip: Path):
    v = Video(two_scene_clip)
    f0 = v.get(0)
    assert f0.shape == (240, 320, 3)
    assert f0.dtype == np.uint8
    # First scene was blue in BGR ⇒ R-channel should be small, B-channel large.
    # decord returns RGB so check accordingly.
    r, g, b = f0[..., 0].mean(), f0[..., 1].mean(), f0[..., 2].mean()
    assert b > r, f"expected blue-dominant first frame, got R={r:.1f} G={g:.1f} B={b:.1f}"


def test_iter_frames_with_stride(two_scene_clip: Path):
    v = Video(two_scene_clip)
    seen = list(v.iter_frames(stride=2))
    assert len(seen) == 30
    assert all(s[0] % 2 == 0 for s in seen)
    assert seen[0][1].shape == (240, 320, 3)


def test_detect_two_scenes(two_scene_clip: Path):
    scenes = detect_scenes(two_scene_clip, content_threshold=27.0, min_scene_len_frames=4)
    assert len(scenes) >= 2, f"expected at least 2 scenes, got {len(scenes)}"
    # Adjacent scenes should tile the whole video without gaps.
    for a, b in zip(scenes, scenes[1:]):
        assert a.end_frame == b.start_frame, (a, b)
    assert scenes[0].start_frame == 0
    assert scenes[-1].end_frame >= 30


def test_detect_scenes_returns_at_least_one_for_static_clip(tmp_path: Path):
    # A clip with no cuts: 30 frames of identical blue.
    p = tmp_path / "static.mp4"
    h, w, fps = 200, 200, 24
    writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    blue = np.full((h, w, 3), (200, 60, 30), dtype=np.uint8)
    for _ in range(30):
        writer.write(blue)
    writer.release()
    scenes = detect_scenes(p)
    assert len(scenes) == 1
    assert scenes[0].start_frame == 0
    assert scenes[0].end_frame >= 30
