"""Character identification via CCIP (Contrastive Character Image Pretraining).

CCIP is purpose-built for outfit- and pose-invariant anime character identity.
Reference images are embedded once at boot; each tracklet is scored by sampling
N evenly-spaced frames, embedding the body crop, and computing the minimum CCIP
distance to any reference. Per-tracklet score is the median of per-frame minima.

Note on the original two-stream design: a separate face-recognition stream was
specified, but no anime-specific face *embedder* is available in standard
libraries (imgutils provides face *detection* only). CCIP on a body crop already
encodes face / hair / eye signature heavily, so a single CCIP stream over the
person bbox covers the same ground robustly. A face-targeted secondary stream
can be added later if E2E verification shows precision gaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image

from neme_anima.config import IdentifyConfig
from neme_anima.detect import Detection
from neme_anima.track import Tracklet
from neme_anima.video import Video


class Verdict(str, Enum):
    KEEP_HIGH = "keep_high"
    KEEP_MEDIUM = "keep_medium"
    REJECT = "reject"


@dataclass(frozen=True)
class TrackletScore:
    scene_idx: int
    tracklet_id: int
    median_distance: float
    per_sample_distances: tuple[float, ...]  # one per sampled frame
    sampled_frame_idxs: tuple[int, ...]
    verdict: Verdict


def _crop_rgb(frame_rgb: np.ndarray, det: Detection) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    x1 = max(0, det.x1)
    y1 = max(0, det.y1)
    x2 = min(w, det.x2)
    y2 = min(h, det.y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame_rgb[y1:y2, x1:x2]


class Identifier:
    """Holds reference embeddings and scores tracklets against them."""

    def __init__(self, *, ref_paths: list[Path], cfg: IdentifyConfig):
        self.cfg = cfg
        self._ref_features: list[np.ndarray] = []
        self._ref_paths: list[Path] = []
        self._load_refs(ref_paths)

    def _load_refs(self, paths: list[Path]) -> None:
        from imgutils.metrics import ccip_extract_feature

        if not paths:
            raise ValueError("No reference images provided")
        for p in paths:
            p = Path(p)
            img = Image.open(p).convert("RGB")
            feat = ccip_extract_feature(img)
            self._ref_features.append(feat)
            self._ref_paths.append(p)

    @property
    def num_references(self) -> int:
        return len(self._ref_features)

    def reference_paths(self) -> tuple[Path, ...]:
        return tuple(self._ref_paths)

    def reference_features(self) -> list[np.ndarray]:
        """Return reference CCIP feature vectors. Used by frame_select.py for ranking."""
        return list(self._ref_features)

    def distance(self, crop_rgb: np.ndarray) -> float:
        """Minimum CCIP distance from a single body-crop to any reference."""
        from imgutils.metrics import ccip_difference, ccip_extract_feature

        if crop_rgb.size == 0 or min(crop_rgb.shape[:2]) < 8:
            return float("inf")
        feat = ccip_extract_feature(Image.fromarray(crop_rgb))
        dists = [ccip_difference(feat, r) for r in self._ref_features]
        return float(min(dists))

    def score_tracklet(self, tracklet: Tracklet, video: Video) -> TrackletScore:
        """Sample evenly across the tracklet, embed each, return median distance."""
        from imgutils.metrics import ccip_batch_extract_features, ccip_difference

        n_samples = min(self.cfg.sample_frames_per_tracklet, tracklet.num_frames)
        if n_samples <= 0:
            return TrackletScore(
                scene_idx=tracklet.scene_idx,
                tracklet_id=tracklet.tracklet_id,
                median_distance=float("inf"),
                per_sample_distances=(),
                sampled_frame_idxs=(),
                verdict=Verdict.REJECT,
            )

        # Indices into tracklet.items, evenly spaced.
        positions = np.linspace(0, tracklet.num_frames - 1, n_samples).astype(int)
        sampled_items = [tracklet.items[i] for i in positions]
        sampled_frame_idxs = tuple(it.frame_idx for it in sampled_items)

        # Batch-fetch frames via decord, then crop each.
        frames = video.get_batch(list(sampled_frame_idxs))
        crops_pil = []
        for it, frame in zip(sampled_items, frames):
            crop = _crop_rgb(frame, it.detection)
            if crop.size == 0 or min(crop.shape[:2]) < 8:
                crops_pil.append(None)
            else:
                crops_pil.append(Image.fromarray(crop))

        valid = [c for c in crops_pil if c is not None]
        per_sample: list[float] = []
        if valid:
            # One batched ONNX forward pass for all samples at once.
            feats = ccip_batch_extract_features(valid)
            valid_iter = iter(feats)
            for c in crops_pil:
                if c is None:
                    per_sample.append(float("inf"))
                    continue
                feat = next(valid_iter)
                per_sample.append(
                    float(min(ccip_difference(feat, r) for r in self._ref_features))
                )
        else:
            per_sample = [float("inf")] * len(crops_pil)

        median = float(np.median(per_sample)) if per_sample else float("inf")
        verdict = self._classify(median)
        return TrackletScore(
            scene_idx=tracklet.scene_idx,
            tracklet_id=tracklet.tracklet_id,
            median_distance=median,
            per_sample_distances=tuple(per_sample),
            sampled_frame_idxs=sampled_frame_idxs,
            verdict=verdict,
        )

    def _classify(self, distance: float) -> Verdict:
        if distance <= self.cfg.body_max_distance_strict:
            return Verdict.KEEP_HIGH
        if distance <= self.cfg.body_max_distance_loose:
            return Verdict.KEEP_MEDIUM
        return Verdict.REJECT


@dataclass(frozen=True)
class RoutedTrackletScore:
    """Per-tracklet routing decision in a multi-character extraction.

    ``character_slug`` is the winning character's slug (lowest median CCIP
    distance) or ``None`` when no character met the loose threshold. ``score``
    is the winning character's :class:`TrackletScore`. ``per_character`` keeps
    the full table around for diagnostics — the WS broadcaster surfaces it so
    the user can see why a tracklet went to A instead of B.
    """
    character_slug: str | None
    score: TrackletScore
    per_character: dict[str, TrackletScore]


class MultiCharacterRouter:
    """Identifies a tracklet against every character's references and routes
    it to the best-matching one (or rejects it).

    Composition of N :class:`Identifier` instances rather than a redesign of
    the single-character one — the per-character contract is unchanged, and
    a project with one character produces output identical (modulo the
    routing wrapper) to the pre-multi-character pipeline.

    Empty character lists in the input map are silently skipped — a project
    can carry a character with zero refs (e.g. a placeholder added before
    the user uploads ref images), and that character simply doesn't compete
    in the routing.
    """

    def __init__(
        self,
        *,
        refs_by_slug: dict[str, list[Path]],
        cfg: IdentifyConfig,
    ) -> None:
        self.cfg = cfg
        self._identifiers: dict[str, Identifier] = {}
        for slug, paths in refs_by_slug.items():
            if not paths:
                continue
            self._identifiers[slug] = Identifier(ref_paths=paths, cfg=cfg)

    @property
    def slugs(self) -> tuple[str, ...]:
        return tuple(self._identifiers.keys())

    def reference_features(self, slug: str) -> list[np.ndarray]:
        return self._identifiers[slug].reference_features()

    def route_tracklet(self, tracklet: Tracklet, video: Video) -> RoutedTrackletScore:
        """Score the tracklet against every character; return the winner.

        Lowest median distance wins. If the winner's verdict is REJECT (no
        character met the loose threshold), ``character_slug`` is ``None``
        and the caller should treat the tracklet as rejected.
        """
        if not self._identifiers:
            # No character has refs — synthesize a reject so the caller
            # doesn't need a separate "no characters configured" branch.
            empty_score = TrackletScore(
                scene_idx=tracklet.scene_idx, tracklet_id=tracklet.tracklet_id,
                median_distance=float("inf"), per_sample_distances=(),
                sampled_frame_idxs=(), verdict=Verdict.REJECT,
            )
            return RoutedTrackletScore(
                character_slug=None, score=empty_score, per_character={},
            )

        scores: dict[str, TrackletScore] = {
            slug: ident.score_tracklet(tracklet, video)
            for slug, ident in self._identifiers.items()
        }
        winner_slug = min(scores, key=lambda s: scores[s].median_distance)
        winner_score = scores[winner_slug]
        if winner_score.verdict == Verdict.REJECT:
            return RoutedTrackletScore(
                character_slug=None, score=winner_score, per_character=scores,
            )
        return RoutedTrackletScore(
            character_slug=winner_slug, score=winner_score, per_character=scores,
        )
