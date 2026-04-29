"""Output directory layout, metadata, and on-disk cache for rerun mode.

Layout produced under `out_root/<video_stem>/`:

    kept/        <name>.png + <name>.txt   (one .txt per .png, kohya tags)
    rejected/    <name>.png                (low-confidence; no tags)
    metadata.json
    thresholds.json
    cache/
        run.json              — video path, fps, frame count, refs
        scenes.parquet        — (scene_idx, start_frame, end_frame)
        tracklets.parquet     — per-(scene, tracklet, frame) detection rows

The cache lets ``rerun`` skip detection + tracking (the slowest stages) and only
re-do identification / frame-selection / cropping / tagging with new thresholds.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from neme_extractor.config import Thresholds
from neme_extractor.detect import Detection, DetectionKind
from neme_extractor.track import TrackedDetection, Tracklet
from neme_extractor.video import Scene


@dataclass
class FrameRecord:
    """One row in metadata.json, for traceability of every kept / rejected image."""
    filename: str
    kept: bool
    scene_idx: int
    tracklet_id: int
    frame_idx: int
    timestamp_seconds: float
    bbox: tuple[int, int, int, int]
    ccip_distance: float
    sharpness: float
    visibility: float
    aspect: float
    score: float


def _safe_name(scene_idx: int, tracklet_id: int, frame_idx: int) -> str:
    return f"s{scene_idx:03d}_t{tracklet_id:03d}_f{frame_idx:06d}"


class OutputWriter:
    def __init__(self, root: Path, video_stem: str):
        self.dir = root / video_stem
        self.kept_dir = self.dir / "kept"
        self.rejected_dir = self.dir / "rejected"
        self.cache_dir = self.dir / "cache"
        for d in (self.dir, self.kept_dir, self.rejected_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._records: list[FrameRecord] = []

    # ------------------------------------------------------------------ images

    def write_kept(
        self,
        record: FrameRecord,
        image_rgb: np.ndarray,
        tag_text: str,
    ) -> Path:
        path = self.kept_dir / f"{record.filename}.png"
        Image.fromarray(image_rgb).save(path)
        path.with_suffix(".txt").write_text(tag_text + "\n", encoding="utf-8")
        self._records.append(record)
        return path

    def write_rejected(
        self,
        record: FrameRecord,
        image_rgb: np.ndarray,
    ) -> Path:
        path = self.rejected_dir / f"{record.filename}.png"
        Image.fromarray(image_rgb).save(path)
        self._records.append(record)
        return path

    @staticmethod
    def filename_for(scene_idx: int, tracklet_id: int, frame_idx: int) -> str:
        return _safe_name(scene_idx, tracklet_id, frame_idx)

    # ---------------------------------------------------------------- metadata

    def flush_metadata(self) -> Path:
        path = self.dir / "metadata.json"
        path.write_text(
            json.dumps([asdict(r) for r in self._records], indent=2)
        )
        return path

    def write_thresholds(self, thresholds: Thresholds) -> Path:
        path = self.dir / "thresholds.json"
        thresholds.to_json(path)
        return path

    def read_thresholds(self) -> Thresholds:
        return Thresholds.from_json(self.dir / "thresholds.json")

    # -------------------------------------------------------------- run header

    def write_run_header(
        self,
        video_path: Path,
        fps: float,
        num_frames: int,
        refs: list[Path],
    ) -> Path:
        path = self.cache_dir / "run.json"
        path.write_text(json.dumps({
            "video_path": str(video_path.resolve()),
            "fps": float(fps),
            "num_frames": int(num_frames),
            "refs": [str(p.resolve()) for p in refs],
        }, indent=2))
        return path

    def read_run_header(self) -> dict:
        return json.loads((self.cache_dir / "run.json").read_text())

    # ----------------------------------------------------------------- caches

    def write_scenes(self, scenes: list[Scene]) -> Path:
        df = pd.DataFrame([{
            "scene_idx": s.index,
            "start_frame": s.start_frame,
            "end_frame": s.end_frame,
        } for s in scenes])
        path = self.cache_dir / "scenes.parquet"
        df.to_parquet(path, index=False)
        return path

    def read_scenes(self) -> list[Scene]:
        df = pd.read_parquet(self.cache_dir / "scenes.parquet")
        return [
            Scene(index=int(r.scene_idx),
                  start_frame=int(r.start_frame),
                  end_frame=int(r.end_frame))
            for r in df.itertuples()
        ]

    def write_tracklets(self, tracklets: list[Tracklet]) -> Path:
        rows = []
        for t in tracklets:
            for it in t.items:
                d = it.detection
                rows.append({
                    "scene_idx": t.scene_idx,
                    "tracklet_id": t.tracklet_id,
                    "frame_idx": it.frame_idx,
                    "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                    "score": d.score,
                    "label": d.label,
                })
        df = pd.DataFrame(rows)
        path = self.cache_dir / "tracklets.parquet"
        df.to_parquet(path, index=False)
        return path

    def read_tracklets(self) -> list[Tracklet]:
        df = pd.read_parquet(self.cache_dir / "tracklets.parquet")
        if df.empty:
            return []
        out: list[Tracklet] = []
        for (scene_idx, tracklet_id), grp in df.groupby(["scene_idx", "tracklet_id"]):
            grp = grp.sort_values("frame_idx")
            items = tuple(
                TrackedDetection(
                    scene_idx=int(scene_idx),
                    tracklet_id=int(tracklet_id),
                    frame_idx=int(r.frame_idx),
                    detection=Detection(
                        kind=DetectionKind.PERSON,
                        x1=int(r.x1), y1=int(r.y1), x2=int(r.x2), y2=int(r.y2),
                        label=str(r.label), score=float(r.score),
                    ),
                )
                for r in grp.itertuples()
            )
            out.append(Tracklet(scene_idx=int(scene_idx),
                                tracklet_id=int(tracklet_id), items=items))
        out.sort(key=lambda t: (t.scene_idx, t.tracklet_id))
        return out
