"""Pick the best 1-3 frames per tracklet (image-quality only — no CCIP).

By the time a tracklet reaches this stage it has already passed identification
(``identify.py``), so every frame in it is the target character. Ranking within
a confirmed tracklet only needs image-quality signals: bbox visibility,
sharpness (Laplacian variance), and aspect-ratio sanity. This is dramatically
faster than per-frame CCIP scoring and quality of picks is not affected.

For long tracklets we cap the candidate pool at ``cfg.candidate_cap`` evenly-
spaced frames (we only ever pick 1-3 anyway). Dedup uses minimum frame distance
between picks, replacing the old CCIP-similarity dedup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neme_extractor.config import FrameSelectConfig
from neme_extractor.quality import aspect_ratio_score, bbox_visibility, sharpness
from neme_extractor.track import Tracklet
from neme_extractor.video import Video


@dataclass(frozen=True)
class FramePick:
    """A single frame chosen for export."""
    scene_idx: int
    tracklet_id: int
    frame_idx: int
    detection_bbox: tuple[int, int, int, int]
    score: float
    sharpness: float
    visibility: float
    aspect: float
    ccip_distance: float  # 0.0 here; kept for metadata-schema compatibility


def _crop_rgb(frame_rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame_rgb[y1:y2, x1:x2]


def select_frames(
    tracklet: Tracklet,
    video: Video,
    ref_features: list[np.ndarray] | None,  # accepted for API compat; ignored now
    cfg: FrameSelectConfig,
) -> list[FramePick]:
    """Return up to top-K best frames from the tracklet, deduped by frame distance."""
    if not tracklet.items:
        return []

    duration = tracklet.duration_seconds(video.fps)
    if duration < cfg.short_tracklet_seconds:
        top_k = cfg.top_k_short
    elif duration >= cfg.long_tracklet_seconds:
        top_k = cfg.top_k_long
    else:
        span = cfg.long_tracklet_seconds - cfg.short_tracklet_seconds
        frac = (duration - cfg.short_tracklet_seconds) / span if span > 0 else 0.0
        top_k = int(round(cfg.top_k_short + frac * (cfg.top_k_long - cfg.top_k_short)))
        top_k = max(cfg.top_k_short, min(cfg.top_k_long, top_k))

    # Down-sample candidate pool for long tracklets — we only ever keep 1–3.
    n = len(tracklet.items)
    if n > cfg.candidate_cap:
        positions = np.linspace(0, n - 1, cfg.candidate_cap).astype(int)
        candidates = [tracklet.items[i] for i in positions]
    else:
        candidates = list(tracklet.items)

    # Batch-fetch the candidate frames in one decord call.
    frame_idxs = [it.frame_idx for it in candidates]
    frames = video.get_batch(frame_idxs)

    # Score each candidate.
    rows: list[tuple[float, int, FramePick]] = []
    sharps: list[float] = []
    bboxes: list[tuple[int, int, int, int]] = []
    viss: list[float] = []
    ars: list[float] = []
    for it, frame in zip(candidates, frames):
        bbox = (it.detection.x1, it.detection.y1, it.detection.x2, it.detection.y2)
        crop = _crop_rgb(frame, bbox)
        sharps.append(sharpness(crop))
        viss.append(bbox_visibility(bbox, frame_w=frame.shape[1], frame_h=frame.shape[0]))
        ars.append(aspect_ratio_score(bbox))
        bboxes.append(bbox)

    sh_arr = np.array(sharps, dtype=np.float64)
    sh_max = float(sh_arr.max()) if sh_arr.size and sh_arr.max() > 0 else 1.0

    for i, it in enumerate(candidates):
        norm_sh = sharps[i] / sh_max
        score = 0.45 * viss[i] + 0.20 * ars[i] + 0.35 * norm_sh
        pick = FramePick(
            scene_idx=tracklet.scene_idx,
            tracklet_id=tracklet.tracklet_id,
            frame_idx=it.frame_idx,
            detection_bbox=bboxes[i],
            score=score,
            sharpness=sharps[i],
            visibility=viss[i],
            aspect=ars[i],
            ccip_distance=0.0,
        )
        rows.append((score, i, pick))

    rows.sort(key=lambda x: -x[0])

    # Greedy dedup: skip a candidate if it's within `dedup_min_frame_gap` frames
    # of one already kept.
    picks: list[FramePick] = []
    for _, _, pick in rows:
        if len(picks) >= top_k:
            break
        if any(abs(pick.frame_idx - kept.frame_idx) < cfg.dedup_min_frame_gap
               for kept in picks):
            continue
        picks.append(pick)
    return picks
