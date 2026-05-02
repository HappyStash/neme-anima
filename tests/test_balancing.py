"""Tests for balancing.py — per-character training-set repeat multipliers."""

from __future__ import annotations

from pathlib import Path

from neme_anima.balancing import (
    compute_character_balancing,
    effective_multiplier_for,
)
from neme_anima.storage.metadata import FrameRecord, MetadataLog
from neme_anima.storage.project import DEFAULT_CHARACTER_SLUG, Project


def _add_kept(project: Project, *, filename: str, character_slug: str) -> None:
    """Append a kept frame metadata row. The on-disk image isn't required
    for balancing — we only count frames via the log."""
    MetadataLog(project.metadata_path).append(FrameRecord(
        filename=filename, kept=True,
        scene_idx=0, tracklet_id=0, frame_idx=0,
        timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
        ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
        score=0.9, video_stem="ep01", character_slug=character_slug,
    ))


def test_single_character_returns_one(tmp_path: Path):
    """One character with N frames balances to multiply=1.0 — there's
    nothing to balance against, but every character must produce a row."""
    project = Project.create(tmp_path / "p", name="solo")
    for i in range(50):
        _add_kept(project, filename=f"f_{i}", character_slug=DEFAULT_CHARACTER_SLUG)
    rows = compute_character_balancing(project=project)
    assert len(rows) == 1
    assert rows[0].character_slug == DEFAULT_CHARACTER_SLUG
    assert rows[0].frame_count == 50
    assert rows[0].auto_multiply == 1.0
    assert rows[0].effective_multiply == 1.0


def test_two_characters_inverse_frequency(tmp_path: Path):
    """Twice as many frames for A as for B → B's multiplier compensates so
    the trainer sees roughly equal exposure per epoch."""
    project = Project.create(tmp_path / "p", name="duo")
    project.add_character(name="Mio")
    for i in range(100):
        _add_kept(project, filename=f"yui_{i}", character_slug=DEFAULT_CHARACTER_SLUG)
    for i in range(50):
        _add_kept(project, filename=f"mio_{i}", character_slug="mio")

    rows = compute_character_balancing(project=project)
    by_slug = {r.character_slug: r for r in rows}
    assert by_slug[DEFAULT_CHARACTER_SLUG].frame_count == 100
    assert by_slug["mio"].frame_count == 50
    # Yui has more than the mean (75) → multiply floors at 1.0.
    # Mio has less than the mean → multiply > 1.0 (specifically 75/50 = 1.5).
    assert by_slug[DEFAULT_CHARACTER_SLUG].auto_multiply == 1.0
    assert by_slug["mio"].auto_multiply == 1.5


def test_clamps_to_max_when_one_character_is_tiny(tmp_path: Path):
    """When one character has very few frames the auto formula could blow
    up; it must clamp at max_multiply so the trainer doesn't see absurd
    repeats. 1 frame vs 1000 → auto would be 500.5; clamp to 10."""
    project = Project.create(tmp_path / "p", name="show")
    project.add_character(name="Tiny")
    for i in range(1000):
        _add_kept(project, filename=f"big_{i}", character_slug=DEFAULT_CHARACTER_SLUG)
    _add_kept(project, filename="tiny_0", character_slug="tiny")
    rows = compute_character_balancing(project=project, max_multiply=10.0)
    by_slug = {r.character_slug: r for r in rows}
    assert by_slug["tiny"].auto_multiply == 10.0


def test_zero_frames_returns_floor(tmp_path: Path):
    """A character with no kept frames still gets a row (UI shows the
    empty state) with multiply=floor — the value is moot since there's
    nothing for the trainer to repeat."""
    project = Project.create(tmp_path / "p", name="show")
    project.add_character(name="Empty")
    rows = compute_character_balancing(project=project, min_multiply=1.0)
    by_slug = {r.character_slug: r for r in rows}
    assert by_slug["empty"].frame_count == 0
    assert by_slug["empty"].auto_multiply == 1.0


def test_manual_override_replaces_auto(tmp_path: Path):
    """Setting Character.multiply > 0 takes precedence over the auto
    formula — the UI surfaces 'auto' AND 'effective' so the user can
    see they've overridden one character without leaving auto for others."""
    project = Project.create(tmp_path / "p", name="duo")
    project.add_character(name="Mio")
    for i in range(100):
        _add_kept(project, filename=f"yui_{i}", character_slug=DEFAULT_CHARACTER_SLUG)
    for i in range(50):
        _add_kept(project, filename=f"mio_{i}", character_slug="mio")
    project.characters[0].multiply = 2.5  # manual override on Yui
    project.save()

    rows = compute_character_balancing(project=project)
    yui = next(r for r in rows if r.character_slug == DEFAULT_CHARACTER_SLUG)
    mio = next(r for r in rows if r.character_slug == "mio")
    assert yui.manual_multiply == 2.5
    assert yui.effective_multiply == 2.5
    # Auto for Yui still computed (1.0 since Yui has more than mean), but
    # the effective multiplier honours the override.
    assert yui.auto_multiply == 1.0
    # Mio retains auto.
    assert mio.manual_multiply == 0.0
    assert mio.effective_multiply == mio.auto_multiply


def test_rejected_frames_excluded_from_count(tmp_path: Path):
    """Last-write-wins: a frame whose latest record is rejected must not
    contribute to the training-corpus count. Otherwise dedup-rejected
    frames would inflate Yui's count and skew the auto multiplier."""
    project = Project.create(tmp_path / "p", name="solo")
    _add_kept(project, filename="keeper", character_slug=DEFAULT_CHARACTER_SLUG)
    # Write a kept record first, then a rejection.
    MetadataLog(project.metadata_path).append(FrameRecord(
        filename="rejected_later", kept=True,
        scene_idx=0, tracklet_id=0, frame_idx=0,
        timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
        ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
        score=0.9, video_stem="ep01",
        character_slug=DEFAULT_CHARACTER_SLUG,
    ))
    MetadataLog(project.metadata_path).append(FrameRecord(
        filename="rejected_later", kept=False,
        scene_idx=0, tracklet_id=0, frame_idx=0,
        timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
        ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
        score=0.9, video_stem="ep01",
        character_slug=DEFAULT_CHARACTER_SLUG,
    ))
    rows = compute_character_balancing(project=project)
    assert rows[0].frame_count == 1


def test_moved_frames_counted_under_new_owner(tmp_path: Path):
    """A frame moved from A to B via the bulk-move endpoint should count
    under B only — last-write-wins applies symmetrically across slugs."""
    project = Project.create(tmp_path / "p", name="duo")
    project.add_character(name="Mio")
    _add_kept(project, filename="moved", character_slug=DEFAULT_CHARACTER_SLUG)
    _add_kept(project, filename="moved", character_slug="mio")  # the move
    rows = compute_character_balancing(project=project)
    by_slug = {r.character_slug: r for r in rows}
    assert by_slug[DEFAULT_CHARACTER_SLUG].frame_count == 0
    assert by_slug["mio"].frame_count == 1


def test_effective_multiplier_for_unknown_slug_returns_one(tmp_path: Path):
    """A metadata row pointing at a character that's been deleted must
    not crash the staging pass — return 1.0 (no balancing) and let the
    user clean up via the Unsorted filter on Frames."""
    project = Project.create(tmp_path / "p", name="solo")
    assert effective_multiplier_for(project, "ghost") == 1.0


def test_rows_ordered_by_project_characters(tmp_path: Path):
    """Rows match project.characters order so the UI table renders in the
    same sequence the user sees in the character switcher strip."""
    project = Project.create(tmp_path / "p", name="trio")
    project.add_character(name="Mio")
    project.add_character(name="Ritsu")
    rows = compute_character_balancing(project=project)
    assert [r.character_slug for r in rows] == [DEFAULT_CHARACTER_SLUG, "mio", "ritsu"]
