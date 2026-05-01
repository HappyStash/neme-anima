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
