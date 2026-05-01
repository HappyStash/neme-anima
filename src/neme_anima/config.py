"""Configuration: thresholds, paths, model IDs. Loadable from / serialisable to JSON."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


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
class Thresholds:
    scene: SceneConfig = field(default_factory=SceneConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    track: TrackConfig = field(default_factory=TrackConfig)
    identify: IdentifyConfig = field(default_factory=IdentifyConfig)
    frame_select: FrameSelectConfig = field(default_factory=FrameSelectConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    tag: TagConfig = field(default_factory=TagConfig)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "Thresholds":
        data = json.loads(path.read_text())
        return cls(
            scene=SceneConfig(**data.get("scene", {})),
            detect=DetectConfig(**data.get("detect", {})),
            track=TrackConfig(**data.get("track", {})),
            identify=IdentifyConfig(**data.get("identify", {})),
            frame_select=FrameSelectConfig(**data.get("frame_select", {})),
            crop=CropConfig(**data.get("crop", {})),
            tag=TagConfig(**{**data.get("tag", {}),
                            "exclude_tags": tuple(data.get("tag", {}).get("exclude_tags", ()))}),
        )
