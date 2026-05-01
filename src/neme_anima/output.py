"""Project-rooted output writer.

Saves kept/rejected images into the project's unified kept/ or rejected/ folder
with a <video_stem>__ filename prefix, and appends one MetadataLog record per
saved frame so the UI / CLI can stream traceability data without scanning the
whole filesystem.

Per-video parquet caches (scenes, tracklets) live under
<project>/output/cache/<video_stem>/ and are read/written by the helpers below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from neme_anima.detect import Detection, DetectionKind
from neme_anima.storage.metadata import FrameRecord, MetadataLog
from neme_anima.storage.project import Project
from neme_anima.track import TrackedDetection, Tracklet
from neme_anima.video import Scene


def _safe_name(video_stem: str, scene_idx: int, tracklet_id: int, frame_idx: int) -> str:
    return f"{video_stem}__s{scene_idx:03d}_t{tracklet_id:03d}_f{frame_idx:06d}"


class OutputWriter:
    def __init__(self, *, project: Project, video_stem: str):
        self.project = project
        self.video_stem = video_stem
        self._metadata = MetadataLog(project.metadata_path)
        self._cache_dir = project.cache_dir_for(video_stem)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def filename_for(*, video_stem: str, scene_idx: int, tracklet_id: int, frame_idx: int) -> str:
        return _safe_name(video_stem, scene_idx, tracklet_id, frame_idx)

    # ---------------- images ----------------

    def write_kept(self, record: FrameRecord, image_rgb: np.ndarray, tag_text: str) -> Path:
        path = self.project.kept_dir / f"{record.filename}.png"
        Image.fromarray(image_rgb).save(path)
        path.with_suffix(".txt").write_text(tag_text + "\n", encoding="utf-8")
        self._metadata.append(record)
        return path

    def write_kept_image(self, record: FrameRecord, image_rgb: np.ndarray) -> Path:
        """Like ``write_kept`` but writes an empty ``.txt`` next to the image —
        used when tagging is deferred to a later pipeline stage so the user
        can review and delete frames before paying the tagger cost.
        """
        path = self.project.kept_dir / f"{record.filename}.png"
        Image.fromarray(image_rgb).save(path)
        path.with_suffix(".txt").write_text("\n", encoding="utf-8")
        self._metadata.append(record)
        return path

    def write_rejected(self, record: FrameRecord, image_rgb: np.ndarray) -> Path:
        path = self.project.rejected_dir / f"{record.filename}.png"
        Image.fromarray(image_rgb).save(path)
        self._metadata.append(record)
        return path

    # ---------------- caches ----------------

    def write_scenes(self, scenes: list[Scene]) -> Path:
        df = pd.DataFrame([{
            "scene_idx": s.index,
            "start_frame": s.start_frame,
            "end_frame": s.end_frame,
        } for s in scenes])
        path = self._cache_dir / "scenes.parquet"
        df.to_parquet(path, index=False)
        return path

    def read_scenes(self) -> list[Scene]:
        df = pd.read_parquet(self._cache_dir / "scenes.parquet")
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
                    "scene_idx": t.scene_idx, "tracklet_id": t.tracklet_id,
                    "frame_idx": it.frame_idx,
                    "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                    "score": d.score, "label": d.label,
                })
        df = pd.DataFrame(rows)
        path = self._cache_dir / "tracklets.parquet"
        df.to_parquet(path, index=False)
        return path

    def read_tracklets(self) -> list[Tracklet]:
        df = pd.read_parquet(self._cache_dir / "tracklets.parquet")
        if df.empty:
            return []
        out: list[Tracklet] = []
        for (scene_idx, tracklet_id), grp in df.groupby(["scene_idx", "tracklet_id"]):
            grp = grp.sort_values("frame_idx")
            items = tuple(
                TrackedDetection(
                    scene_idx=int(scene_idx), tracklet_id=int(tracklet_id),
                    frame_idx=int(r.frame_idx),
                    detection=Detection(
                        kind=DetectionKind.PERSON,
                        x1=int(r.x1), y1=int(r.y1), x2=int(r.x2), y2=int(r.y2),
                        label=str(r.label), score=float(r.score),
                    ),
                ) for r in grp.itertuples()
            )
            out.append(Tracklet(scene_idx=int(scene_idx),
                                tracklet_id=int(tracklet_id), items=items))
        out.sort(key=lambda t: (t.scene_idx, t.tracklet_id))
        return out
