"""Unit tests for the prefix-scoped output wipe used by run_rerun.

The rerun path deletes only files belonging to ONE video, identified by the
``<video_stem>__`` filename prefix. The trailing double-underscore is what
prevents collisions between e.g. ``ep01`` and ``ep01ext``.
"""

from __future__ import annotations

from pathlib import Path

from neme_anima.pipeline import _wipe_outputs_for_stem
from neme_anima.storage.project import Project


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_wipe_targets_only_matching_prefix(tmp_path: Path):
    project = Project.create(tmp_path / "p", name="p")
    # Three videos with prefixes that share leading characters.
    _touch(project.kept_dir / "ep01__s000_t001_f000010.png")
    _touch(project.kept_dir / "ep01__s000_t001_f000010.txt")
    _touch(project.kept_dir / "ep01ext__s000_t001_f000010.png")
    _touch(project.kept_dir / "ep01ext__s000_t001_f000010.txt")
    _touch(project.kept_dir / "ep02__s000_t001_f000010.png")
    _touch(project.rejected_dir / "ep01__s000_t099_f000999.png")
    _touch(project.rejected_dir / "ep02__s000_t099_f000999.png")

    _wipe_outputs_for_stem(project, "ep01")

    kept_remaining = sorted(p.name for p in project.kept_dir.iterdir())
    rejected_remaining = sorted(p.name for p in project.rejected_dir.iterdir())

    # ep01 files gone; ep01ext (different stem) and ep02 untouched.
    assert kept_remaining == [
        "ep01ext__s000_t001_f000010.png",
        "ep01ext__s000_t001_f000010.txt",
        "ep02__s000_t001_f000010.png",
    ]
    assert rejected_remaining == ["ep02__s000_t099_f000999.png"]


def test_wipe_handles_missing_directories(tmp_path: Path):
    """If output dirs don't exist (fresh project / never extracted), wipe is a no-op."""
    project = Project.create(tmp_path / "p", name="p")
    # Manually remove the dirs that Project.create made, to simulate a missing-state.
    import shutil
    shutil.rmtree(project.kept_dir)
    shutil.rmtree(project.rejected_dir)
    _wipe_outputs_for_stem(project, "ep01")  # must not raise


def test_run_extract_wipes_prior_stem_outputs_before_new_writes(
    tmp_path: Path, monkeypatch,
):
    """Regression: a re-Run on a video that was already extracted must
    replace the prior outputs, not append to them. Without the wipe,
    leftover frames from a previous Run (different refs / different
    characters / different scan thresholds) survive into the new tag
    pass and silently pollute the dataset.

    Verified by pre-seeding stale files matching the stem, monkey-
    patching scene detection to bail early, calling run_extract, and
    asserting the stale files are gone — proving the wipe ran in
    setup, BEFORE the pipeline tried to write anything new.
    """
    import pytest
    from neme_anima import pipeline as pipeline_mod
    from neme_anima.pipeline import run_extract

    # Make a real (tiny) clip so the Video() open in setup succeeds.
    import cv2
    import numpy as np
    clip = tmp_path / "ep01.mp4"
    writer = cv2.VideoWriter(
        str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 24, (160, 120),
    )
    for _ in range(8):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    project = Project.create(tmp_path / "p", name="p")
    # Stamp a ref so the refs_by_slug check passes — we just need the
    # path to be a file. Tagging never runs because we'll bail at scenes.
    fake_ref = tmp_path / "ref.png"
    fake_ref.write_bytes(b"\x89PNG\r\n\x1a\n")
    project.add_ref(fake_ref)
    project.add_source(clip)

    # Stale files from a "prior Extract" — these are exactly what the
    # bug left behind: phantom frames that survived a re-Run with
    # different settings.
    stale_png = project.kept_dir / "ep01__s000_t000_f000010.png"
    stale_txt = project.kept_dir / "ep01__s000_t000_f000010.txt"
    stale_rejected = project.rejected_dir / "ep01__s000_t000_f000020.png"
    _touch(stale_png)
    _touch(stale_txt)
    _touch(stale_rejected)
    # A frame from a different video must NOT be wiped.
    other_png = project.kept_dir / "ep02__s000_t000_f000010.png"
    _touch(other_png)

    # Make scene detection bail so we don't need GPU/CCIP/YOLO. By the
    # time this raises, the wipe has already run in setup.
    def _explode(*a, **kw):
        raise RuntimeError("test-induced bail after setup")
    monkeypatch.setattr(pipeline_mod, "detect_scenes", _explode)

    with pytest.raises(RuntimeError, match="test-induced bail"):
        run_extract(project=project, source_idx=0)

    # ep01's stale outputs are gone (wipe happened in setup) — confirms
    # the fix. ep02's untouched, proving the wipe stays prefix-scoped.
    assert not stale_png.exists()
    assert not stale_txt.exists()
    assert not stale_rejected.exists()
    assert other_png.exists()
