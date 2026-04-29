"""Smoke tests for detect.py.

The model-call test triggers a one-time weights download from HuggingFace and
runs the detector on a generated noise frame: it must not crash and must
return a tuple. Empty results are expected (the detector is anime-trained, noise
has no anime characters).
"""

from __future__ import annotations

import numpy as np
import pytest

from neme_extractor.detect import (
    Detection,
    DetectionKind,
    Detector,
    FrameDetections,
    assign_face_to_person,
)


def test_detector_instantiates():
    d = Detector()
    assert d.person_level == "m"
    assert d.face_level == "s"


def test_assign_face_to_person_picks_smallest_enclosing():
    persons = (
        Detection(DetectionKind.PERSON, 0, 0, 1000, 1000, "person", 0.9),
        Detection(DetectionKind.PERSON, 100, 100, 300, 400, "person", 0.85),
        Detection(DetectionKind.PERSON, 600, 0, 800, 200, "person", 0.7),
    )
    face = Detection(DetectionKind.FACE, 150, 120, 200, 170, "face", 0.95)
    pick = assign_face_to_person(face, persons)
    assert pick is not None
    # Face center is at (175, 145). Both the big 0..1000 box and the 100..300/100..400
    # box contain it; the smaller one should win.
    assert pick.x1 == 100 and pick.y1 == 100


def test_assign_face_to_person_returns_none_when_no_overlap():
    persons = (
        Detection(DetectionKind.PERSON, 600, 0, 800, 200, "person", 0.7),
    )
    face = Detection(DetectionKind.FACE, 150, 120, 200, 170, "face", 0.95)
    assert assign_face_to_person(face, persons) is None


@pytest.mark.gpu
def test_detect_runs_on_noise_frame_without_crashing():
    """Verifies the model loads and runs end-to-end. Empty results are fine."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    d = Detector(person_level="n", face_level="n")  # smallest models for test speed
    persons = d.detect_persons(frame)
    faces = d.detect_faces(frame)
    assert isinstance(persons, tuple)
    assert isinstance(faces, tuple)
    fd = d.detect_frame(0, frame)
    assert isinstance(fd, FrameDetections)
    assert fd.frame_idx == 0
