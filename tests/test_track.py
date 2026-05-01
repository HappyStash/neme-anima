"""Smoke tests for track.py — synthesize fake detections moving across frames,
verify ByteTrack assigns consistent tracklet IDs and they reset across scenes.
"""

from __future__ import annotations

from neme_anima.config import TrackConfig
from neme_anima.detect import Detection, DetectionKind, FrameDetections
from neme_anima.track import track_scene


def _person(x: int, y: int, w: int = 80, h: int = 200, score: float = 0.9) -> Detection:
    return Detection(
        kind=DetectionKind.PERSON,
        x1=x, y1=y, x2=x + w, y2=y + h,
        label="person", score=score,
    )


def _frame(idx: int, persons: list[Detection]) -> FrameDetections:
    return FrameDetections(frame_idx=idx, persons=tuple(persons), faces=())


def test_track_one_person_steady_across_frames():
    """A single person walks 4 px/frame across 20 frames → one tracklet."""
    cfg = TrackConfig(min_tracklet_len=3)
    frames = [_frame(i, [_person(100 + 4 * i, 200)]) for i in range(20)]
    tracklets = track_scene(scene_idx=0, scene_frames=frames, config=cfg)
    assert len(tracklets) == 1
    t = tracklets[0]
    assert t.scene_idx == 0
    assert t.num_frames >= 18  # ByteTrack typically takes 1-2 frames to confirm
    assert t.start_frame >= 0


def test_track_two_separated_people():
    """Two people on opposite sides of the frame → two tracklets, distinct IDs."""
    cfg = TrackConfig(min_tracklet_len=3)
    frames = [
        _frame(i, [_person(100 + 2 * i, 200), _person(800 - 2 * i, 200)])
        for i in range(20)
    ]
    tracklets = track_scene(scene_idx=0, scene_frames=frames, config=cfg)
    assert len(tracklets) == 2
    ids = {t.tracklet_id for t in tracklets}
    assert len(ids) == 2


def test_track_scene_index_propagates():
    cfg = TrackConfig(min_tracklet_len=2)
    frames = [_frame(i, [_person(100, 200)]) for i in range(10)]
    tracklets = track_scene(scene_idx=7, scene_frames=frames, config=cfg)
    assert tracklets
    assert all(t.scene_idx == 7 for t in tracklets)
    assert all(item.scene_idx == 7 for t in tracklets for item in t.items)


def test_track_handles_frames_with_no_detections():
    """Mid-scene drop in detections shouldn't crash the tracker."""
    cfg = TrackConfig(min_tracklet_len=2)
    frames = []
    for i in range(20):
        if 8 <= i <= 10:
            frames.append(_frame(i, []))
        else:
            frames.append(_frame(i, [_person(100 + 2 * i, 200)]))
    tracklets = track_scene(scene_idx=0, scene_frames=frames, config=cfg)
    # Track may be a single tracklet (recovered) or two short ones — either is fine,
    # we just want no crash and at least one tracklet.
    assert len(tracklets) >= 1


def test_track_tracklet_ids_independent_across_scenes():
    """Running track_scene twice with different scene_idx keeps both IDs valid;
    the values may collide (each scene gets its own ByteTrack), but downstream code
    differentiates by (scene_idx, tracklet_id)."""
    cfg = TrackConfig(min_tracklet_len=2)
    frames_a = [_frame(i, [_person(100 + 2 * i, 200)]) for i in range(10)]
    frames_b = [_frame(100 + i, [_person(100 + 2 * i, 200)]) for i in range(10)]
    a = track_scene(scene_idx=0, scene_frames=frames_a, config=cfg)
    b = track_scene(scene_idx=1, scene_frames=frames_b, config=cfg)
    assert a and b
    keys_a = {(t.scene_idx, t.tracklet_id) for t in a}
    keys_b = {(t.scene_idx, t.tracklet_id) for t in b}
    assert keys_a.isdisjoint(keys_b)  # composite key always differs
