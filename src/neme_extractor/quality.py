"""Per-frame quality metrics: sharpness, visibility, mask connectedness.

These run cheap and are used by frame_select.py to rank frames within a tracklet.
Mask-based metrics are optional — they're only useful once a segmentation mask
is available (i.e., after the crop step). The frame ranker can run with bbox-
based metrics alone and use mask metrics only when re-ranking finalists.
"""

from __future__ import annotations

import cv2
import numpy as np


def sharpness(crop_rgb: np.ndarray) -> float:
    """Laplacian variance: higher = sharper. Robust to brightness, sensitive to blur."""
    if crop_rgb.size == 0 or min(crop_rgb.shape[:2]) < 3:
        return 0.0
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def bbox_visibility(
    bbox: tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    edge_margin: int = 2,
) -> float:
    """Fraction of bbox NOT touching the frame edge.

    Returns 1.0 when the bbox sits comfortably inside the frame, less when any
    side abuts the border (indicating the character is being chopped at the edge).
    """
    x1, y1, x2, y2 = bbox
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    if bw == 0 or bh == 0:
        return 0.0
    chopped = 0
    if x1 <= edge_margin:
        chopped += 1
    if y1 <= edge_margin:
        chopped += 1
    if x2 >= frame_w - edge_margin:
        chopped += 1
    if y2 >= frame_h - edge_margin:
        chopped += 1
    # Each chopped side reduces visibility by 0.25.
    return max(0.0, 1.0 - 0.25 * chopped)


def aspect_ratio_score(bbox: tuple[int, int, int, int]) -> float:
    """Penalize extreme aspect ratios (e.g., a sliver-thin or pancake-flat bbox).

    Anime full-body shots are typically taller than wide (~0.4-0.6 width/height).
    Score is 1 at ratio 0.5, falling toward 0 as ratio diverges.
    """
    x1, y1, x2, y2 = bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    ratio = w / h
    # Quadratic falloff centered at 0.5; broadly tolerant from 0.2 to 1.5.
    return max(0.0, 1.0 - ((ratio - 0.5) ** 2) * 1.5)


def mask_connectedness(mask: np.ndarray) -> float:
    """Fraction of mask area that lies in its largest connected component.

    A clean character mask is one connected blob; a fragmented mask
    (character split by an obstacle, or two characters bleeding into one box)
    scores low.
    """
    if mask.size == 0 or mask.sum() == 0:
        return 0.0
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return 0.0
    # stats[0] is the background component; pick the largest non-bg.
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return 0.0
    return float(areas.max() / binary.sum())
