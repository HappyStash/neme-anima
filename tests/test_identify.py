"""Smoke tests for identify.py.

CCIP is trained on anime characters; on synthesized non-anime images the
embeddings cluster very tightly (everything looks like 'no character'). These
tests therefore only verify plumbing — loading refs, the distance contract,
threshold classification, and that score_tracklet executes end-to-end on a
synthetic clip. End-to-end matching quality is verified separately.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from neme_extractor.config import IdentifyConfig
from neme_extractor.detect import Detection, DetectionKind
from neme_extractor.identify import Identifier, Verdict
from neme_extractor.track import TrackedDetection, Tracklet
from neme_extractor.video import Video


@pytest.fixture
def refs_dir(tmp_path: Path) -> Path:
    """A folder with two synthetic reference images."""
    d = tmp_path / "refs"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        arr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(arr).save(d / f"ref_{i}.png")
    return d


def test_identifier_loads_references(refs_dir: Path):
    ident = Identifier(refs_dir, IdentifyConfig())
    assert ident.num_references == 2
    paths = ident.reference_paths()
    assert len(paths) == 2
    assert all(p.exists() for p in paths)


def test_identifier_rejects_empty_refs_dir(tmp_path: Path):
    empty = tmp_path / "empty_refs"
    empty.mkdir()
    with pytest.raises(ValueError, match="No reference images"):
        Identifier(empty, IdentifyConfig())


def test_distance_self_is_near_zero(refs_dir: Path):
    """Distance from a reference image to itself should be ~0."""
    ident = Identifier(refs_dir, IdentifyConfig())
    # The first ref's pixels.
    arr = np.array(Image.open(ident.reference_paths()[0]).convert("RGB"))
    assert ident.distance(arr) < 1e-3


def test_distance_handles_tiny_or_empty_crops(refs_dir: Path):
    ident = Identifier(refs_dir, IdentifyConfig())
    assert ident.distance(np.zeros((0, 0, 3), dtype=np.uint8)) == float("inf")
    assert ident.distance(np.zeros((4, 4, 3), dtype=np.uint8)) == float("inf")


def test_classify_thresholds():
    cfg = IdentifyConfig(body_max_distance_strict=0.10, body_max_distance_loose=0.20)
    ident = Identifier.__new__(Identifier)  # don't load refs for this unit test
    ident.cfg = cfg
    assert ident._classify(0.05) == Verdict.KEEP_HIGH
    assert ident._classify(0.10) == Verdict.KEEP_HIGH
    assert ident._classify(0.15) == Verdict.KEEP_MEDIUM
    assert ident._classify(0.20) == Verdict.KEEP_MEDIUM
    assert ident._classify(0.21) == Verdict.REJECT
    assert ident._classify(float("inf")) == Verdict.REJECT


@pytest.fixture
def synthetic_clip(tmp_path: Path) -> Path:
    """Tiny 1-second clip — used to exercise score_tracklet's video reads."""
    p = tmp_path / "clip.mp4"
    h, w, fps = 240, 320, 24
    writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(24):
        f = np.full((h, w, 3), (60, 80, 200), dtype=np.uint8)
        writer.write(f)
    writer.release()
    return p


def test_score_tracklet_executes(refs_dir: Path, synthetic_clip: Path):
    """End-to-end plumbing: build a fake tracklet, score it, get a finite verdict."""
    cfg = IdentifyConfig(sample_frames_per_tracklet=3,
                         body_max_distance_strict=0.10,
                         body_max_distance_loose=0.20)
    ident = Identifier(refs_dir, cfg)
    video = Video(synthetic_clip)

    items = []
    for fi in range(0, 24, 4):
        det = Detection(
            kind=DetectionKind.PERSON,
            x1=50, y1=40, x2=200, y2=200,
            label="person", score=0.9,
        )
        items.append(TrackedDetection(scene_idx=0, tracklet_id=1, frame_idx=fi, detection=det))
    tracklet = Tracklet(scene_idx=0, tracklet_id=1, items=tuple(items))

    score = ident.score_tracklet(tracklet, video)
    assert score.scene_idx == 0
    assert score.tracklet_id == 1
    assert len(score.per_sample_distances) == 3
    assert all(np.isfinite(d) for d in score.per_sample_distances)
    assert score.verdict in (Verdict.KEEP_HIGH, Verdict.KEEP_MEDIUM, Verdict.REJECT)
