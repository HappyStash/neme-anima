"""Smoke tests for crop.py — framing math + isnetis runs end-to-end."""

from __future__ import annotations

import numpy as np
import pytest

from neme_anima.config import CropConfig
from neme_anima.crop import crop_frame


def test_crop_preserves_aspect_and_resizes_to_longest_side():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    cfg = CropConfig(longest_side=512, pad_ratio=0.10)
    # 200x600 box (h>w → tall); padded by 10% then resized so max side = 512.
    bbox = (800, 200, 1000, 800)  # 200 wide, 600 tall
    out = crop_frame(frame, bbox, cfg, compute_mask=False)
    assert out.image_rgb.dtype == np.uint8
    h, w = out.image_rgb.shape[:2]
    assert max(h, w) == 512
    assert h > w  # aspect preserved (originally tall)


def test_crop_clips_to_frame_bounds():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    cfg = CropConfig(longest_side=128, pad_ratio=0.5)
    # Box partly off-screen on the left and top.
    bbox = (-20, -10, 100, 200)
    out = crop_frame(frame, bbox, cfg, compute_mask=False)
    sx1, sy1, sx2, sy2 = out.source_bbox_in_frame
    assert sx1 >= 0 and sy1 >= 0
    assert sx2 <= 320 and sy2 <= 240


def test_crop_pixels_match_source_pixels():
    """The crop must be a direct sample of the original frame, not a recolored copy."""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    # Draw a distinctive marker block in original pixels.
    frame[100:120, 100:120] = (255, 0, 0)
    cfg = CropConfig(longest_side=400, pad_ratio=0.0)  # no pad, no resize beyond original
    bbox = (95, 95, 125, 125)  # 30x30 → resized to 400x400
    out = crop_frame(frame, bbox, cfg, compute_mask=False)
    # Center pixel of resized crop must be red (no recolor / mask cutout occurred).
    cy, cx = out.image_rgb.shape[0] // 2, out.image_rgb.shape[1] // 2
    r, g, b = out.image_rgb[cy, cx]
    assert int(r) > 200 and int(g) < 50 and int(b) < 50


@pytest.mark.gpu
def test_crop_with_mask_runs_end_to_end():
    """Ensure isnetis mask path returns a binary mask aligned with the image."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    cfg = CropConfig(longest_side=256, pad_ratio=0.10)
    out = crop_frame(frame, (100, 50, 300, 400), cfg, compute_mask=True)
    assert out.mask is not None
    assert out.mask.shape == out.image_rgb.shape[:2]
    assert out.mask.dtype == np.uint8
    assert set(np.unique(out.mask).tolist()).issubset({0, 1})
