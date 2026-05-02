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

    The scoped-wipe path adds nuance: prior frames are deleted only when
    they belong to a character that's *active* in the new run (has at
    least one effective ref for this source). Rejected files always
    wipe. The seeded data here belongs to the project's only character
    so it's active and gets wiped; this test pins the post-scoped-wipe
    behaviour for the historical "Run again with the same character
    set" path.
    """
    import cv2
    import numpy as np
    import pytest

    from neme_anima import pipeline as pipeline_mod
    from neme_anima.pipeline import run_extract
    from neme_anima.storage.metadata import FrameRecord, MetadataLog
    from neme_anima.storage.project import DEFAULT_CHARACTER_SLUG

    # Make a real (tiny) clip so the Video() open in setup succeeds.
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
    # different settings. They belong to the default character, which
    # is the only character in this project and therefore active here.
    stale_png = project.kept_dir / "ep01__s000_t000_f000010.png"
    stale_txt = project.kept_dir / "ep01__s000_t000_f000010.txt"
    stale_rejected = project.rejected_dir / "ep01__s000_t000_f000020.png"
    _touch(stale_png)
    _touch(stale_txt)
    _touch(stale_rejected)
    log = MetadataLog(project.metadata_path)
    log.append(FrameRecord(
        filename="ep01__s000_t000_f000010", kept=True,
        scene_idx=0, tracklet_id=0, frame_idx=10,
        timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
        ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
        score=0.9, video_stem="ep01",
        character_slug=DEFAULT_CHARACTER_SLUG,
    ))
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

    # ep01's tracked frames are gone — they belong to the default
    # character which is active in this run. ep02 untouched (different
    # stem); rejected always wiped.
    assert not stale_png.exists()
    assert not stale_txt.exists()
    assert not stale_rejected.exists()
    assert other_png.exists()


def test_run_extract_preserves_frames_owned_by_inactive_character(
    tmp_path: Path, monkeypatch,
):
    """Scoped wipe — the user-visible promise. Two characters: Yui has
    refs, Mio has no refs (or all opted-out). Pre-seed prior frames for
    BOTH characters. After Extract, Yui's frames are wiped (active),
    Mio's frames are preserved (no effective refs in this run).

    Plus rejected files always wipe (diagnostic samples), and untracked
    files always preserve (no way to attribute them, safer to keep).
    """
    import cv2
    import numpy as np
    import pytest

    from neme_anima import pipeline as pipeline_mod
    from neme_anima.pipeline import run_extract
    from neme_anima.storage.metadata import FrameRecord, MetadataLog
    from neme_anima.storage.project import DEFAULT_CHARACTER_SLUG

    clip = tmp_path / "ep01.mp4"
    writer = cv2.VideoWriter(
        str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 24, (160, 120),
    )
    for _ in range(8):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    project = Project.create(tmp_path / "p", name="p")
    project.add_character(name="Mio")
    # Only the default character (Yui) gets a ref → only Yui is active.
    fake_ref = tmp_path / "ref.png"
    fake_ref.write_bytes(b"\x89PNG\r\n\x1a\n")
    project.add_ref(fake_ref)  # routes to default character
    project.add_source(clip)

    # Seed:
    #   - one Yui frame (active, will be wiped)
    #   - one Mio frame (inactive, preserved)
    #   - one untracked frame (no metadata, preserved as conservative
    #     default — we can't attribute it to any character)
    #   - one rejected file (always wiped — diagnostic, not curation)
    yui_png = project.kept_dir / "ep01__s000_t000_f000010.png"
    mio_png = project.kept_dir / "ep01__s001_t001_f000020.png"
    untracked_png = project.kept_dir / "ep01__manual_drop.png"
    rejected_png = project.rejected_dir / "ep01__s000_t099_f000099.png"
    for p in (yui_png, mio_png, untracked_png, rejected_png):
        _touch(p)
    log = MetadataLog(project.metadata_path)
    log.append(FrameRecord(
        filename="ep01__s000_t000_f000010", kept=True,
        scene_idx=0, tracklet_id=0, frame_idx=10,
        timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
        ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
        score=0.9, video_stem="ep01",
        character_slug=DEFAULT_CHARACTER_SLUG,
    ))
    log.append(FrameRecord(
        filename="ep01__s001_t001_f000020", kept=True,
        scene_idx=1, tracklet_id=1, frame_idx=20,
        timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
        ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
        score=0.9, video_stem="ep01", character_slug="mio",
    ))

    def _explode(*a, **kw):
        raise RuntimeError("test-induced bail after setup")
    monkeypatch.setattr(pipeline_mod, "detect_scenes", _explode)

    with pytest.raises(RuntimeError, match="test-induced bail"):
        run_extract(project=project, source_idx=0)

    assert not yui_png.exists()       # active → wiped
    assert mio_png.exists()           # inactive → preserved
    assert untracked_png.exists()     # no metadata → preserved
    assert not rejected_png.exists()  # rejected always wiped


def test_legacy_full_wipe_when_preserve_owned_by_is_none(tmp_path: Path):
    """The wipe-everything contract is still callable directly — the
    rerun and extract paths now route through preserve_owned_by, but
    callers that need the historical behaviour (e.g. a future "force
    full wipe" UI button or a migration utility) can pass None."""
    project = Project.create(tmp_path / "p", name="p")
    _touch(project.kept_dir / "ep01__a.png")
    _touch(project.kept_dir / "ep01__a.txt")
    _touch(project.kept_dir / "ep02__a.png")
    _wipe_outputs_for_stem(project, "ep01")  # no preserve set → wipe all
    assert not (project.kept_dir / "ep01__a.png").exists()
    assert not (project.kept_dir / "ep01__a.txt").exists()
    assert (project.kept_dir / "ep02__a.png").exists()
