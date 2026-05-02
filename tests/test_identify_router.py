"""Tests for MultiCharacterRouter — the multi-character routing layer.

Pure-routing logic is exercised here without loading CCIP: we patch the
``score_tracklet`` method on each underlying ``Identifier`` so the router
sees pre-canned distances and we can assert the routing decision.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image

from neme_anima.config import IdentifyConfig
from neme_anima.detect import Detection, DetectionKind
from neme_anima.identify import (
    Identifier,
    MultiCharacterRouter,
    TrackletScore,
    Verdict,
)
from neme_anima.track import TrackedDetection, Tracklet
from neme_anima.video import Video


@pytest.fixture
def two_ref_sets(tmp_path: Path) -> dict[str, list[Path]]:
    """Two single-image ref sets, one per synthetic 'character'."""
    rng = np.random.default_rng(0)
    sets: dict[str, list[Path]] = {}
    for slug in ("yui", "mio"):
        arr = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
        p = tmp_path / f"{slug}.png"
        Image.fromarray(arr).save(p)
        sets[slug] = [p]
    return sets


@pytest.fixture
def synthetic_clip(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    h, w, fps = 120, 160, 24
    writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(24):
        writer.write(np.full((h, w, 3), 100, dtype=np.uint8))
    writer.release()
    return p


def _stub_score(median: float, verdict: Verdict) -> TrackletScore:
    return TrackletScore(
        scene_idx=0, tracklet_id=1,
        median_distance=median,
        per_sample_distances=(median,) * 3,
        sampled_frame_idxs=(0, 4, 8),
        verdict=verdict,
    )


def _make_tracklet() -> Tracklet:
    items = []
    for fi in range(0, 12, 2):
        det = Detection(DetectionKind.PERSON, 10, 10, 100, 100, "person", 0.9)
        items.append(TrackedDetection(scene_idx=0, tracklet_id=1,
                                       frame_idx=fi, detection=det))
    return Tracklet(scene_idx=0, tracklet_id=1, items=tuple(items))


def test_route_picks_lowest_median_distance(
    two_ref_sets: dict[str, list[Path]], synthetic_clip: Path,
):
    """Router routes a tracklet to the character with the lowest median CCIP
    distance among those passing the loose threshold. The losing character's
    score is still preserved in ``per_character`` for diagnostics."""
    cfg = IdentifyConfig(body_max_distance_strict=0.10,
                         body_max_distance_loose=0.20)
    router = MultiCharacterRouter(refs_by_slug=two_ref_sets, cfg=cfg)
    video = Video(synthetic_clip)
    tracklet = _make_tracklet()

    # Yui: closer match (0.05); Mio: borderline (0.18). Both KEEP, Yui wins.
    side_effects = {
        "yui": _stub_score(0.05, Verdict.KEEP_HIGH),
        "mio": _stub_score(0.18, Verdict.KEEP_MEDIUM),
    }

    def fake_score(self, tracklet, video):
        slug = next(s for s, ident in router._identifiers.items() if ident is self)
        return side_effects[slug]

    with patch.object(Identifier, "score_tracklet", new=fake_score):
        routed = router.route_tracklet(tracklet, video)

    assert routed.character_slug == "yui"
    assert routed.score.median_distance == 0.05
    assert set(routed.per_character) == {"yui", "mio"}


def test_route_rejects_when_all_above_loose_threshold(
    two_ref_sets: dict[str, list[Path]], synthetic_clip: Path,
):
    """If neither character meets the loose threshold the tracklet is rejected,
    even though one is closer than the other. Routing is not "best of bad
    options" — it requires the winner to actually pass."""
    cfg = IdentifyConfig(body_max_distance_strict=0.10,
                         body_max_distance_loose=0.20)
    router = MultiCharacterRouter(refs_by_slug=two_ref_sets, cfg=cfg)
    tracklet = _make_tracklet()
    video = Video(synthetic_clip)

    side_effects = {
        "yui": _stub_score(0.30, Verdict.REJECT),
        "mio": _stub_score(0.40, Verdict.REJECT),
    }

    def fake_score(self, tracklet, video):
        slug = next(s for s, ident in router._identifiers.items() if ident is self)
        return side_effects[slug]

    with patch.object(Identifier, "score_tracklet", new=fake_score):
        routed = router.route_tracklet(tracklet, video)

    assert routed.character_slug is None
    assert routed.score.verdict == Verdict.REJECT


def test_route_skips_characters_with_empty_refs(
    two_ref_sets: dict[str, list[Path]], synthetic_clip: Path,
):
    """A character with zero refs (e.g. just-created, no images uploaded yet)
    must not crash routing — the router silently drops it from the per-char
    table and routes among the rest."""
    refs = dict(two_ref_sets)
    refs["empty"] = []  # carries no refs
    cfg = IdentifyConfig()
    router = MultiCharacterRouter(refs_by_slug=refs, cfg=cfg)

    assert "empty" not in router.slugs
    assert set(router.slugs) == {"yui", "mio"}


def test_route_with_no_characters_returns_reject():
    """An entirely empty refs map yields a synthetic REJECT — callers don't
    need a separate "no characters configured" branch."""
    cfg = IdentifyConfig()
    router = MultiCharacterRouter(refs_by_slug={}, cfg=cfg)
    tracklet = _make_tracklet()
    routed = router.route_tracklet(tracklet, video=None)  # type: ignore[arg-type]
    assert routed.character_slug is None
    assert routed.score.verdict == Verdict.REJECT
    assert routed.per_character == {}


def test_single_character_routing_identical_to_legacy(
    two_ref_sets: dict[str, list[Path]], synthetic_clip: Path,
):
    """A project with one character should route every tracklet to that
    character whenever its score passes — the router's behaviour collapses
    cleanly to the pre-multi-character pipeline."""
    cfg = IdentifyConfig(body_max_distance_strict=0.10,
                         body_max_distance_loose=0.20)
    router = MultiCharacterRouter(
        refs_by_slug={"only": two_ref_sets["yui"]}, cfg=cfg,
    )
    tracklet = _make_tracklet()
    video = Video(synthetic_clip)

    fake = _stub_score(0.05, Verdict.KEEP_HIGH)

    with patch.object(Identifier, "score_tracklet", return_value=fake):
        routed = router.route_tracklet(tracklet, video)

    assert routed.character_slug == "only"
    assert list(routed.per_character) == ["only"]
