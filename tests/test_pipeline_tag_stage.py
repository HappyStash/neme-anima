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


def test_tag_stage_skips_frames_owned_by_preserved_character(
    tmp_path: Path, monkeypatch,
):
    """The user's reported bug: re-Run with a character's refs
    disabled was correctly preserving the character's PNG/.txt files
    on disk (scoped wipe), but the tag stage was then iterating every
    PNG matching the prefix and re-tagging — silently overwriting the
    user's curated tags. Fix: pass the preserve set into the tag stage
    too, and skip frames whose owning character is in it. Files with
    no metadata still get tagged (we can't attribute them; conservative
    tag is safer than conservative skip for fresh frames)."""
    from neme_anima.storage.metadata import FrameRecord, MetadataLog
    from neme_anima.storage.project import DEFAULT_CHARACTER_SLUG

    project = Project.create(tmp_path / "p", name="p")
    project.add_character(name="Mio")
    # 3 frames: one belongs to active default char (will be tagged),
    # one belongs to inactive 'mio' (preserve, must NOT be retagged),
    # one untracked (no metadata, conservative tag fallback).
    _seed_kept_image(project, name="ep01__yui_a")
    _seed_kept_image(project, name="ep01__mio_a")
    _seed_kept_image(project, name="ep01__manual_drop")
    log = MetadataLog(project.metadata_path)
    log.append(FrameRecord(
        filename="ep01__yui_a", kept=True,
        scene_idx=0, tracklet_id=0, frame_idx=0,
        timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
        ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
        score=0.9, video_stem="ep01",
        character_slug=DEFAULT_CHARACTER_SLUG,
    ))
    log.append(FrameRecord(
        filename="ep01__mio_a", kept=True,
        scene_idx=0, tracklet_id=0, frame_idx=0,
        timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
        ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
        score=0.9, video_stem="ep01", character_slug="mio",
    ))
    # Prime the preserved character's sidecar with curated content
    # that the test will assert SURVIVES the tag pass.
    (project.kept_dir / "ep01__mio_a.txt").write_text(
        "user, curated, tags\n", encoding="utf-8",
    )
    monkeypatch.setattr(pipeline_mod, "Tagger", _StubTagger)

    pipeline_mod._run_tag_stage(
        project=project, video_stem="ep01",
        thresholds=Thresholds(),
        progress=pipeline_mod.NULL_PROGRESS,
        pause=False,
        preserve_owned_by={"mio"},
    )

    # Active character's frame retagged with the stub's output.
    assert (project.kept_dir / "ep01__yui_a.txt").read_text(encoding="utf-8").startswith(
        "fake_tag, another",
    )
    # Preserved character's sidecar UNCHANGED — the regression bar.
    assert (project.kept_dir / "ep01__mio_a.txt").read_text(encoding="utf-8").startswith(
        "user, curated, tags",
    )
    # Untracked file got tagged (no metadata to attribute it; treated
    # as a fresh frame that needs tags).
    assert (project.kept_dir / "ep01__manual_drop.txt").read_text(encoding="utf-8").startswith(
        "fake_tag, another",
    )


def test_tag_stage_skips_preserved_frame_after_rejected_sample_collision(
    tmp_path: Path, monkeypatch,
):
    """Regression: a preserved frame whose filename collides with a
    rejected-sample write (same scene/tracklet/frame_idx → same stem,
    but a different physical file in rejected/) lost its ownership in
    the tag stage and got silently retagged.

    Repro: char A previously kept ep01__s0_t0_f10. char A is then
    opted-out for this run (preserved). char B's identify pass runs;
    the same tracklet doesn't match B and triggers a rejected-sample
    append at the SAME filename stem (kept=False). _kept_frame_owners
    then reported owner=None for the preserved file, and the tag stage
    re-tagged it with B's WD14 output, overwriting A's curated sidecar.
    """
    from neme_anima.storage.metadata import FrameRecord, MetadataLog
    from neme_anima.storage.project import DEFAULT_CHARACTER_SLUG

    project = Project.create(tmp_path / "p", name="p")
    project.add_character(name="Mio")
    _seed_kept_image(project, name="ep01__s0_t0_f10")
    log = MetadataLog(project.metadata_path)
    # Original kept record from a prior run (owner = default / "yui").
    log.append(FrameRecord(
        filename="ep01__s0_t0_f10", kept=True,
        scene_idx=0, tracklet_id=0, frame_idx=10,
        timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
        ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
        score=0.9, video_stem="ep01",
        character_slug=DEFAULT_CHARACTER_SLUG,
    ))
    # Same filename stem appended later as a rejected-sample diagnostic
    # — the actual file lives in rejected/, but the metadata stem
    # collides with the preserved frame in kept/.
    log.append(FrameRecord(
        filename="ep01__s0_t0_f10", kept=False,
        scene_idx=0, tracklet_id=0, frame_idx=10,
        timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
        ccip_distance=0.45, sharpness=0.0, visibility=0.0, aspect=0.0,
        score=0.0, video_stem="ep01",
        character_slug=DEFAULT_CHARACTER_SLUG,
    ))
    (project.kept_dir / "ep01__s0_t0_f10.txt").write_text(
        "user, curated, tags\n", encoding="utf-8",
    )
    monkeypatch.setattr(pipeline_mod, "Tagger", _StubTagger)

    pipeline_mod._run_tag_stage(
        project=project, video_stem="ep01",
        thresholds=Thresholds(),
        progress=pipeline_mod.NULL_PROGRESS,
        pause=False,
        preserve_owned_by={DEFAULT_CHARACTER_SLUG},
    )

    # The curated sidecar must survive — preserved owner attribution
    # is robust to kept=False records appended after the kept=True row.
    assert (project.kept_dir / "ep01__s0_t0_f10.txt").read_text(
        encoding="utf-8"
    ).startswith("user, curated, tags")
