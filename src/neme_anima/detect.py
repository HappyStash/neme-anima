"""Anime person + face detection via DeepGHS imgutils (YOLOv8 fine-tunes)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from PIL import Image

from imgutils.detect import detect_faces, detect_person


class DetectionKind(str, Enum):
    PERSON = "person"
    FACE = "face"


@dataclass(frozen=True)
class Detection:
    """A single detected entity in a frame, in pixel coordinates of that frame."""
    kind: DetectionKind
    x1: int
    y1: int
    x2: int
    y2: int
    label: str           # imgutils returns a class label string ("person", "face", etc.)
    score: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def contains_point(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass(frozen=True)
class FrameDetections:
    """All detections found in a single frame."""
    frame_idx: int
    persons: tuple[Detection, ...]
    faces: tuple[Detection, ...]


class Detector:
    """Anime person + face detector. Uses imgutils YOLOv8 anime fine-tunes.

    Both detectors run on the same frame; faces are kept independently and matched
    to person bboxes downstream (in identify.py).
    """

    def __init__(
        self,
        person_level: Literal["n", "s", "m", "x"] = "m",
        person_version: str = "v1.1",
        person_score_min: float = 0.3,
        face_level: Literal["s", "n"] = "s",
        face_version: str = "v1.4",
        face_score_min: float = 0.25,
    ) -> None:
        self.person_level = person_level
        self.person_version = person_version
        self.person_score_min = person_score_min
        self.face_level = face_level
        self.face_version = face_version
        self.face_score_min = face_score_min

    @staticmethod
    def _to_pil(frame_rgb: np.ndarray) -> Image.Image:
        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8)
        return Image.fromarray(frame_rgb, mode="RGB")

    def detect_persons(self, frame_rgb: np.ndarray) -> tuple[Detection, ...]:
        img = self._to_pil(frame_rgb)
        raw = detect_person(
            img,
            level=self.person_level,
            version=self.person_version,
            conf_threshold=self.person_score_min,
        )
        return tuple(
            Detection(
                kind=DetectionKind.PERSON,
                x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3]),
                label=label,
                score=float(score),
            )
            for (b, label, score) in raw
        )

    def detect_faces(self, frame_rgb: np.ndarray) -> tuple[Detection, ...]:
        img = self._to_pil(frame_rgb)
        raw = detect_faces(
            img,
            level=self.face_level,
            version=self.face_version,
            conf_threshold=self.face_score_min,
        )
        return tuple(
            Detection(
                kind=DetectionKind.FACE,
                x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3]),
                label=label,
                score=float(score),
            )
            for (b, label, score) in raw
        )

    def detect_frame(
        self, frame_idx: int, frame_rgb: np.ndarray, *, with_faces: bool = True
    ) -> FrameDetections:
        return FrameDetections(
            frame_idx=frame_idx,
            persons=self.detect_persons(frame_rgb),
            faces=self.detect_faces(frame_rgb) if with_faces else (),
        )


def assign_face_to_person(
    face: Detection, persons: tuple[Detection, ...]
) -> Detection | None:
    """Return the person bbox most likely to contain this face, or None.

    Heuristic: the face center must lie inside a person bbox; if multiple match,
    pick the smallest-area enclosing box (tightest fit).
    """
    cx = (face.x1 + face.x2) / 2
    cy = (face.y1 + face.y2) / 2
    candidates = [p for p in persons if p.contains_point(cx, cy)]
    if not candidates:
        return None
    return min(candidates, key=lambda p: p.area)
