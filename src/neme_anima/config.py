"""Configuration: thresholds, paths, model IDs. Loadable from / serialisable to JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def _filter_known(dc_cls: type, raw: dict) -> dict:
    """Drop keys that aren't declared on ``dc_cls``.

    Lets ``from_json`` tolerate legacy fields without crashing — e.g.
    ``dedup.enabled`` was a real config in earlier releases and may still
    sit in saved JSON, but it isn't a kwarg the dataclass accepts now.
    """
    declared = {f.name for f in fields(dc_cls)}
    return {k: v for k, v in raw.items() if k in declared}


@dataclass
class SceneConfig:
    threshold: float = 27.0
    min_scene_len_frames: int = 8


@dataclass
class DetectConfig:
    person_score_min: float = 0.35
    face_score_min: float = 0.35
    frame_stride: int = 4              # every Nth frame; 4 @ 24 fps = 6 effective fps
    detect_faces: bool = False         # face stream not used by current matcher; saves ~45% of detect time


@dataclass
class TrackConfig:
    track_thresh: float = 0.25
    match_thresh: float = 0.8
    frame_rate: int = 30
    track_buffer: int = 30
    min_tracklet_len: int = 3  # frames


@dataclass
class IdentifyConfig:
    """CCIP distance thresholds. Lower = more similar; default ~0.178 means 'same character'."""
    body_max_distance_strict: float = 0.15   # below this = high confidence keep
    body_max_distance_loose: float = 0.20    # below this = medium confidence keep
    sample_frames_per_tracklet: int = 5


@dataclass
class FrameSelectConfig:
    short_tracklet_seconds: float = 1.0
    long_tracklet_seconds: float = 5.0
    top_k_short: int = 1
    top_k_long: int = 3
    candidate_cap: int = 20           # for long tracklets, score this many evenly-spaced frames
    dedup_min_frame_gap: int = 4      # picks must be at least this many frames apart


@dataclass
class CropConfig:
    longest_side: int = 1024
    pad_ratio: float = 0.10  # extra padding around mask, as a fraction of bbox size


@dataclass
class TagConfig:
    """WD14 tagging settings. ``model_name`` is the imgutils key
    (e.g. 'EVA02_Large', 'SwinV2_v3'); see imgutils.tagging.wd14.MODEL_NAMES.
    """
    model_name: str = "EVA02_Large"  # SmilingWolf/wd-eva02-large-tagger-v3
    general_threshold: float = 0.35
    character_threshold: float = 0.85
    no_underline: bool = True
    drop_overlap: bool = True
    exclude_tags: tuple[str, ...] = ()


@dataclass
class DedupConfig:
    """Perceptual dedup pass over kept crops using CCIP embeddings.

    Cross-tracklet near-duplicates leak past the in-tracklet frame-gap
    dedup — repeated takes inside a single shot, near-identical poses
    across adjacent cuts. Restricted to a temporal window so dedup only
    collapses *locally* repeated frames: comparing every kept crop to
    every other kept crop across the whole video also collapsed
    legitimately distinct shots that happened to be visually similar
    (a character against the same background twenty minutes apart),
    which is rarely what you want.

    ``lookback_frames`` is the maximum frame_idx delta between two
    kept crops for them to be considered duplicate-eligible — at 24
    fps, the default 1000 covers ~40 seconds, comfortably wider than
    a typical anime shot. Set to 0 to compare across the whole video
    (legacy behaviour, useful for very short clips).

    ``distance_threshold`` defaults to 0.02 — strict enough that only
    crops which are essentially the same image collapse together. The
    looser 0.05 default used previously also caught near-duplicates
    of different poses inside the same shot, which the windowed pass
    can now afford to leave in place since it never compares across
    distant cuts anyway.
    """
    distance_threshold: float = 0.02  # CCIP distance below this = duplicate
    lookback_frames: int = 1000       # max frame_idx delta; 0 = unlimited
    move_to_rejected: bool = True     # False = delete; True = move to rejected/
    embed_batch_size: int = 256       # crops per GPU forward pass during embedding


@dataclass
class Thresholds:
    scene: SceneConfig = field(default_factory=SceneConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    identify: IdentifyConfig = field(default_factory=IdentifyConfig)
    frame_select: FrameSelectConfig = field(default_factory=FrameSelectConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    tag: TagConfig = field(default_factory=TagConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "Thresholds":
        data = json.loads(path.read_text())
        tag_raw = data.get("tag", {})
        return cls(
            scene=SceneConfig(**_filter_known(SceneConfig, data.get("scene", {}))),
            detect=DetectConfig(**_filter_known(DetectConfig, data.get("detect", {}))),
            track=TrackConfig(**_filter_known(TrackConfig, data.get("track", {}))),
            identify=IdentifyConfig(**_filter_known(IdentifyConfig, data.get("identify", {}))),
            frame_select=FrameSelectConfig(
                **_filter_known(FrameSelectConfig, data.get("frame_select", {}))
            ),
            crop=CropConfig(**_filter_known(CropConfig, data.get("crop", {}))),
            tag=TagConfig(**{
                **_filter_known(TagConfig, tag_raw),
                "exclude_tags": tuple(tag_raw.get("exclude_tags", ())),
            }),
            dedup=DedupConfig(**_filter_known(DedupConfig, data.get("dedup", {}))),
        )
