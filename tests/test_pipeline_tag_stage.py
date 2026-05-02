"""Tests for ``_run_tag_stage`` — the stage that pause-before-tag resumes into.

Regression guard for the bug where the stage iterated every PNG in
``kept_dir`` (including ``_crop.png`` derivatives) and wrote a sidecar
for each, producing phantom ``_crop.txt`` files that have no valid
semantics in the data model.

We don't load WD14 here (CPU-only test env). Instead we monkeypatch the
``Tagger`` class to a stub that records which image sources it was
called with and returns deterministic tag text. That's enough to verify
the iteration / sidecar-writing contract without booking the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from neme_anima import pipeline as pipeline_mod
from neme_anima.config import Thresholds
from neme_anima.storage.project import Project


@dataclass
class _StubTagResult:
    text: str = "fake_tag, another"
    general: dict = None  # type: ignore[assignment]
    character: dict = None  # type: ignore[assignment]
    rating: dict = None  # type: ignore[assignment]


class _StubTagger:
    """Records every image source the stage hands it. The recorded list
    drives the assertions about what got tagged."""
    def __init__(self, *args, **kwargs) -> None:
        self.tagged_paths: list[Path] = []

    def tag(self, arr) -> _StubTagResult:  # noqa: ANN001
        return _StubTagResult()


def _seed_kept_image(project: Project, *, name: str) -> Path:
    """Write a real PNG so PIL.Image.open + numpy succeed inside the stage."""
    png = project.kept_dir / f"{name}.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(png)
    # Mirror what write_kept_image does — empty-but-present sidecar.
    png.with_suffix(".txt").write_text("\n", encoding="utf-8")
    return png


def test_tag_stage_skips_crop_derivatives_and_writes_one_sidecar_per_original(
    tmp_path: Path, monkeypatch,
):
    """One PNG (frame_a) has a sibling _crop.png; the stage must produce
    ONE sidecar (at the original's path) — not two (one per file)."""
    project = Project.create(tmp_path / "p", name="p")
    _seed_kept_image(project, name="ep01__a")
    _seed_kept_image(project, name="ep01__a_crop")  # the derivative
    _seed_kept_image(project, name="ep01__b")  # no crop
    monkeypatch.setattr(pipeline_mod, "Tagger", _StubTagger)

    pipeline_mod._run_tag_stage(
        project=project, video_stem="ep01",
        thresholds=Thresholds(),
        progress=pipeline_mod.NULL_PROGRESS,
        pause=False,
    )

    # Originals get a fresh sidecar; the derivative does NOT.
    assert (project.kept_dir / "ep01__a.txt").read_text(encoding="utf-8").startswith(
        "fake_tag, another",
    )
    assert (project.kept_dir / "ep01__b.txt").read_text(encoding="utf-8").startswith(
        "fake_tag, another",
    )
    assert not (project.kept_dir / "ep01__a_crop.txt").exists()


def test_tag_stage_uses_crop_pixels_when_derivative_present(
    tmp_path: Path, monkeypatch,
):
    """When a frame has a crop derivative, the tagger must see the
    derivative's pixels (those are what the trainer trains on), not the
    original's. Captured by inspecting the PIL image the stub opens
    indirectly — we colour the original red and the crop blue, then
    assert the recorded mean channel."""
    project = Project.create(tmp_path / "p", name="p")
    orig = project.kept_dir / "ep01__a.png"
    crop = project.kept_dir / "ep01__a_crop.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(orig)
    Image.new("RGB", (8, 8), (0, 0, 255)).save(crop)
    orig.with_suffix(".txt").write_text("\n", encoding="utf-8")

    captured: list[np.ndarray] = []

    class _CapturingTagger:
        def __init__(self, *a, **kw) -> None: ...
        def tag(self, arr):  # noqa: ANN001
            captured.append(arr.copy())
            return _StubTagResult()

    monkeypatch.setattr(pipeline_mod, "Tagger", _CapturingTagger)
    pipeline_mod._run_tag_stage(
        project=project, video_stem="ep01",
        thresholds=Thresholds(),
        progress=pipeline_mod.NULL_PROGRESS,
        pause=False,
    )
    # One frame tagged → one capture. The captured pixels should be
    # blue-dominant (the crop) not red-dominant (the original).
    assert len(captured) == 1
    arr = captured[0]
    assert arr[..., 2].mean() > arr[..., 0].mean()


def test_tag_stage_sweeps_stale_crop_sidecars_at_start(
    tmp_path: Path, monkeypatch,
):
    """A pre-existing ``_crop.txt`` from an older buggy run must be
    cleaned up on every fresh tag pass — they have no valid semantics
    in the data model and confuse the dataset preview."""
    project = Project.create(tmp_path / "p", name="p")
    _seed_kept_image(project, name="ep01__a")
    _seed_kept_image(project, name="ep01__a_crop")
    # Plant a stray sidecar from a prior buggy run.
    (project.kept_dir / "ep01__a_crop.txt").write_text(
        "stale, tags, from_bug\n", encoding="utf-8",
    )
    monkeypatch.setattr(pipeline_mod, "Tagger", _StubTagger)

    pipeline_mod._run_tag_stage(
        project=project, video_stem="ep01",
        thresholds=Thresholds(),
        progress=pipeline_mod.NULL_PROGRESS,
        pause=False,
    )
    assert not (project.kept_dir / "ep01__a_crop.txt").exists()


def test_tag_stage_only_touches_files_for_this_video_stem(
    tmp_path: Path, monkeypatch,
):
    """Per-video-stem prefix isolation: another video's frames + sidecars
    must NOT be touched by this stage. A previous bug where the stage
    nuked or re-tagged unrelated stems would silently corrupt
    multi-source projects."""
    project = Project.create(tmp_path / "p", name="p")
    _seed_kept_image(project, name="ep01__a")
    _seed_kept_image(project, name="ep02__a")
    # Stale crop sidecar belonging to ep02 — must NOT be swept by an
    # ep01 tag pass.
    (project.kept_dir / "ep02__a_crop.txt").write_text("ep02_stale\n", encoding="utf-8")
    (project.kept_dir / "ep02__a.txt").write_text("ep02_existing\n", encoding="utf-8")
    monkeypatch.setattr(pipeline_mod, "Tagger", _StubTagger)

    pipeline_mod._run_tag_stage(
        project=project, video_stem="ep01",
        thresholds=Thresholds(),
        progress=pipeline_mod.NULL_PROGRESS,
        pause=False,
    )
    # ep02 untouched.
    assert (project.kept_dir / "ep02__a.txt").read_text(encoding="utf-8").strip() == "ep02_existing"
    assert (project.kept_dir / "ep02__a_crop.txt").exists()
    # ep01 retagged.
    assert (project.kept_dir / "ep01__a.txt").read_text(encoding="utf-8").startswith(
        "fake_tag, another",
    )
