"""Video I/O and scene detection.

Wraps decord for fast batched frame reads and scenedetect for shot boundaries.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from scenedetect import ContentDetector, SceneManager, open_video


@dataclass(frozen=True)
class Scene:
    """A continuous shot in the video, half-open frame range [start_frame, end_frame)."""
    index: int
    start_frame: int
    end_frame: int

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame

    def duration_seconds(self, fps: float) -> float:
        return self.num_frames / fps


class Video:
    """Lazy random-access video reader returning RGB uint8 ndarrays."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._vr = VideoReader(str(self.path), ctx=cpu(0))
        self.fps = float(self._vr.get_avg_fps())
        self.num_frames = len(self._vr)

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0

    def __len__(self) -> int:
        return self.num_frames

    def get(self, idx: int) -> np.ndarray:
        """Return a single frame as HxWx3 uint8 RGB."""
        return self._vr[idx].asnumpy()

    def get_batch(self, indices: list[int]) -> np.ndarray:
        """Return frames as NxHxWx3 uint8 RGB."""
        if not indices:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)
        return self._vr.get_batch(indices).asnumpy()

    def iter_frames(
        self,
        start: int = 0,
        end: int | None = None,
        stride: int = 1,
        batch_size: int = 32,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_idx, frame_rgb) for frame_idx in range(start, end, stride),
        reading in batches of ``batch_size`` for throughput.
        """
        if end is None:
            end = self.num_frames
        end = min(end, self.num_frames)
        if stride < 1:
            raise ValueError("stride must be >= 1")
        idxs = list(range(start, end, stride))
        for batch_start in range(0, len(idxs), batch_size):
            batch_idxs = idxs[batch_start: batch_start + batch_size]
            frames = self.get_batch(batch_idxs)
            for fi, frame in zip(batch_idxs, frames):
                yield fi, frame


def detect_scenes(
    video_path: Path,
    *,
    content_threshold: float = 27.0,
    min_scene_len_frames: int = 8,
) -> list[Scene]:
    """Detect shot boundaries with PySceneDetect's ContentDetector.

    Returns scenes as half-open frame ranges. Always returns at least one scene
    spanning the whole video, even if no cuts are detected.
    """
    pys_video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(
        ContentDetector(threshold=content_threshold, min_scene_len=min_scene_len_frames)
    )
    sm.detect_scenes(pys_video, show_progress=False)
    raw = sm.get_scene_list()
    scenes: list[Scene] = []
    if not raw:
        # No cuts found — entire video is one scene.
        v = Video(video_path)
        scenes.append(Scene(index=0, start_frame=0, end_frame=v.num_frames))
        return scenes
    for i, (start_tc, end_tc) in enumerate(raw):
        scenes.append(
            Scene(
                index=i,
                start_frame=int(start_tc.get_frames()),
                end_frame=int(end_tc.get_frames()),
            )
        )
    return scenes
