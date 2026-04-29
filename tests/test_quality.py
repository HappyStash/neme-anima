"""Unit tests for quality metrics."""

from __future__ import annotations

import cv2
import numpy as np

from neme_extractor.quality import (
    aspect_ratio_score,
    bbox_visibility,
    mask_connectedness,
    sharpness,
)


def test_sharpness_higher_for_textured_than_blurred():
    rng = np.random.default_rng(0)
    textured = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(textured, (15, 15), 5).astype(np.uint8)
    assert sharpness(textured) > sharpness(blurred) * 5


def test_sharpness_handles_empty():
    assert sharpness(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0
    assert sharpness(np.zeros((1, 1, 3), dtype=np.uint8)) == 0.0


def test_bbox_visibility_full_inside():
    # bbox solidly inside a 1000x1000 frame
    assert bbox_visibility((100, 100, 300, 700), 1000, 1000) == 1.0


def test_bbox_visibility_one_chop():
    # bbox flush against the left edge
    v = bbox_visibility((0, 100, 200, 700), 1000, 1000)
    assert 0.7 < v < 0.8


def test_bbox_visibility_two_chops():
    # bbox in a corner
    v = bbox_visibility((0, 0, 200, 200), 1000, 1000)
    assert 0.4 < v < 0.6


def test_aspect_ratio_centered_at_half():
    assert aspect_ratio_score((0, 0, 100, 200)) >= 0.99   # ratio = 0.5, ideal
    assert aspect_ratio_score((0, 0, 100, 100)) > 0.5     # square
    assert aspect_ratio_score((0, 0, 1000, 100)) < 0.2    # very wide


def test_mask_connectedness_single_blob():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 30:70] = 1
    assert mask_connectedness(mask) == 1.0


def test_mask_connectedness_two_blobs_equal():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    mask[60:80, 60:80] = 1  # same area as the other blob
    score = mask_connectedness(mask)
    assert 0.45 < score < 0.55


def test_mask_connectedness_empty():
    mask = np.zeros((10, 10), dtype=np.uint8)
    assert mask_connectedness(mask) == 0.0
