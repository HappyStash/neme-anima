"""Per-scene tracking with ByteTrack.

Tracklet IDs are scoped to a single scene: a fresh tracker is constructed for each
scene so identities never persist across hard cuts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import supervision as sv

from neme_extractor.config import TrackConfig
from neme_extractor.detect import Detection, FrameDetections


@dataclass(frozen=True)
class TrackedDetection:
    """A person detection labelled with its tracklet identity."""
    scene_idx: int
    tracklet_id: int
    frame_idx: int
    detection: Detection


@dataclass(frozen=True)
class Tracklet:
    """One continuous appearance of a (presumably single) character within a scene."""
    scene_idx: int
    tracklet_id: int
    items: tuple[TrackedDetection, ...]  # ordered by frame_idx ascending

    @property
    def start_frame(self) -> int:
        return self.items[0].frame_idx

    @property
    def end_frame(self) -> int:
        return self.items[-1].frame_idx

    @property
    def num_frames(self) -> int:
        return len(self.items)

    def duration_seconds(self, fps: float) -> float:
        return (self.end_frame - self.start_frame + 1) / fps if fps > 0 else 0.0


def _persons_to_sv_detections(
    persons: tuple[Detection, ...]
) -> sv.Detections:
    """Build a supervision.Detections object from a tuple of person Detections."""
    if not persons:
        return sv.Detections.empty()
    xyxy = np.array(
        [[p.x1, p.y1, p.x2, p.y2] for p in persons], dtype=np.float32
    )
    conf = np.array([p.score for p in persons], dtype=np.float32)
    class_id = np.zeros(len(persons), dtype=int)  # single class: person
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_id)


def track_scene(
    scene_idx: int,
    scene_frames: list[FrameDetections],
    config: TrackConfig,
) -> list[Tracklet]:
    """Run ByteTrack on the person detections of all frames in one scene.

    Returns a list of tracklets. Tracklet ids are unique within the scene only.
    """
    tracker = sv.ByteTrack(
        track_activation_threshold=config.track_thresh,
        minimum_matching_threshold=config.match_thresh,
        frame_rate=config.frame_rate,
        lost_track_buffer=config.track_buffer,
        minimum_consecutive_frames=1,
    )

    items_by_id: dict[int, list[TrackedDetection]] = {}

    for fd in scene_frames:
        if not fd.persons:
            # Even with no detections, advance the tracker so frames are accounted for.
            tracker.update_with_detections(sv.Detections.empty())
            continue

        sv_dets = _persons_to_sv_detections(fd.persons)
        tracked = tracker.update_with_detections(sv_dets)

        if tracked.tracker_id is None or len(tracked) == 0:
            continue

        # Map each tracked output back to a source Detection by IoU.
        for k in range(len(tracked)):
            tid = int(tracked.tracker_id[k])
            tx1, ty1, tx2, ty2 = (float(v) for v in tracked.xyxy[k])
            # Find the source detection with best IoU (typically a perfect match
            # since ByteTrack keeps the input box, just labels it).
            best, best_iou = None, -1.0
            for det in fd.persons:
                iou = _iou((tx1, ty1, tx2, ty2),
                          (det.x1, det.y1, det.x2, det.y2))
                if iou > best_iou:
                    best, best_iou = det, iou
            if best is None:
                continue
            items_by_id.setdefault(tid, []).append(
                TrackedDetection(
                    scene_idx=scene_idx,
                    tracklet_id=tid,
                    frame_idx=fd.frame_idx,
                    detection=best,
                )
            )

    tracklets: list[Tracklet] = []
    for tid, items in items_by_id.items():
        items.sort(key=lambda x: x.frame_idx)
        if len(items) < config.min_tracklet_len:
            continue
        tracklets.append(
            Tracklet(scene_idx=scene_idx, tracklet_id=tid, items=tuple(items))
        )
    tracklets.sort(key=lambda t: (t.start_frame, t.tracklet_id))
    return tracklets


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0
