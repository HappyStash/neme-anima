"""Unit tests for the dedup module.

The pure helpers (group-finding, keeper-selection, move/delete) cover the
algorithm without needing CCIP. The end-to-end ``dedup_kept_for_video`` path
needs the GPU model group and is exercised separately.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from neme_anima.config import DedupConfig, Thresholds
from neme_anima.dedup import (
    _move_or_delete,
    dedup_kept_for_video,
    find_duplicate_groups,
    select_keepers,
)
from neme_anima.storage.metadata import FrameRecord, MetadataLog
from neme_anima.storage.project import Project


def test_find_duplicate_groups_empty_matrix_returns_empty():
    groups = find_duplicate_groups(np.zeros((0, 0)), threshold=0.1)
    assert groups == []


def test_find_duplicate_groups_no_edges_all_singletons():
    """With every distance well above threshold, each point is its own group."""
    n = 4
    d = np.full((n, n), 1.0)
    np.fill_diagonal(d, 0.0)
    groups = find_duplicate_groups(d, threshold=0.1)
    assert sorted(map(sorted, groups)) == [[0], [1], [2], [3]]


def test_find_duplicate_groups_transitive_merge():
    """0–1 close, 1–2 close, but 0–2 above threshold → all three still merge."""
    d = np.array([
        [0.0, 0.04, 0.30],
        [0.04, 0.0, 0.04],
        [0.30, 0.04, 0.0],
    ])
    groups = find_duplicate_groups(d, threshold=0.05)
    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1, 2]


def test_find_duplicate_groups_strict_less_than():
    """Distance == threshold is NOT a duplicate. Sentinel for 0-tolerance configs."""
    d = np.array([[0.0, 0.05], [0.05, 0.0]])
    groups = find_duplicate_groups(d, threshold=0.05)
    assert sorted(map(sorted, groups)) == [[0], [1]]


def test_find_duplicate_groups_lookback_window_blocks_far_pairs():
    """With ``lookback_frames`` set, a near-zero distance pair whose
    frame_idx delta exceeds the window must NOT merge — that's the
    whole point of windowing dedup.

    Three frames: 0 and 2 have an exact-duplicate distance (0.0) but
    are 5000 frames apart in the video; 1 sits between them at frame
    100 with no near-match. Window = 1000 frames → 0 and 2 stay
    distinct (would have merged under the legacy all-pairs algorithm)."""
    d = np.array([
        [0.0, 0.30, 0.0],
        [0.30, 0.0, 0.30],
        [0.0, 0.30, 0.0],
    ])
    frame_indices = [0, 100, 5000]
    groups = find_duplicate_groups(
        d, threshold=0.05,
        frame_indices=frame_indices, lookback_frames=1000,
    )
    assert sorted(map(sorted, groups)) == [[0], [1], [2]]


def test_find_duplicate_groups_lookback_window_keeps_close_pairs():
    """The other side of the contract: a near-duplicate inside the
    window still merges. Two frames 50 apart with distance 0.02 must
    end up in one group when lookback_frames=1000."""
    d = np.array([
        [0.0, 0.02],
        [0.02, 0.0],
    ])
    groups = find_duplicate_groups(
        d, threshold=0.05,
        frame_indices=[0, 50], lookback_frames=1000,
    )
    assert sorted(map(sorted, groups)) == [[0, 1]]


def test_find_duplicate_groups_lookback_zero_disables_window():
    """``lookback_frames=0`` is the explicit "compare everything" sentinel
    — same behaviour as the legacy all-pairs API. Provided as an escape
    hatch for very short clips where the window would isolate every
    frame trivially."""
    d = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    groups = find_duplicate_groups(
        d, threshold=0.05,
        frame_indices=[0, 99999], lookback_frames=0,
    )
    assert sorted(map(sorted, groups)) == [[0, 1]]


def test_find_duplicate_groups_lookback_requires_frame_indices():
    """Asking for windowed dedup without supplying frame_indices is a
    programmer error — fail loud rather than silently dropping the
    window restriction."""
    import pytest
    d = np.zeros((2, 2))
    with pytest.raises(ValueError, match="frame_indices"):
        find_duplicate_groups(d, threshold=0.05, lookback_frames=500)


def test_select_keepers_singletons_always_kept():
    keep, drop = select_keepers([[0], [1], [2]], scores=[0.1, 0.5, 0.9])
    assert keep == {0, 1, 2}
    assert drop == set()


def test_select_keepers_picks_highest_score_in_each_group():
    keep, drop = select_keepers(
        [[0, 1, 2], [3, 4]],
        scores=[0.1, 0.9, 0.4, 0.7, 0.3],
    )
    assert keep == {1, 3}
    assert drop == {0, 2, 4}


def test_select_keepers_ties_resolve_to_lowest_index():
    """Determinism: tied scores fall to the lowest index. Two equal-quality
    crops shouldn't oscillate keep/drop between runs."""
    keep, drop = select_keepers([[2, 5, 7]], scores=[0, 0, 0.5, 0, 0, 0.5, 0, 0.5])
    assert keep == {2}
    assert drop == {5, 7}


def test_move_or_delete_moves_png_and_txt(tmp_path: Path):
    """move_to_rejected=True relocates png + sidecar; original directory empties."""
    project_root = tmp_path / "proj"
    project = Project.create(project_root, name="t")

    png = project.kept_dir / "vid__s000_t000_f000000.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(png)
    png.with_suffix(".txt").write_text("a, b\n", encoding="utf-8")
    crop_png = png.with_name(f"{png.stem}_crop.png")
    crop_json = png.with_suffix(".crop.json")
    crop_png.write_bytes(b"fake")
    crop_json.write_text("{}", encoding="utf-8")

    _move_or_delete(png, project, move_to_rejected=True)

    assert not png.exists()
    assert not png.with_suffix(".txt").exists()
    # Crop derivatives are dropped entirely — they're outputs of a now-rejected
    # frame and have no value in rejected/.
    assert not crop_png.exists()
    assert not crop_json.exists()
    assert (project.rejected_dir / png.name).exists()
    assert (project.rejected_dir / png.with_suffix(".txt").name).exists()


def test_move_or_delete_delete_mode(tmp_path: Path):
    project_root = tmp_path / "proj"
    project = Project.create(project_root, name="t")

    png = project.kept_dir / "vid__s000_t000_f000000.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(png)
    png.with_suffix(".txt").write_text("a, b\n", encoding="utf-8")

    _move_or_delete(png, project, move_to_rejected=False)

    assert not png.exists()
    assert not png.with_suffix(".txt").exists()
    assert not (project.rejected_dir / png.name).exists()


def test_dedup_no_kept_frames_is_noop(tmp_path: Path):
    """Empty kept_dir for the stem returns a clean zero report — guard against
    empty-directory crashes inside CCIP batch extract. Dedup is always on
    now, so this is the closest thing to a "do-nothing" path."""
    project_root = tmp_path / "proj"
    project = Project.create(project_root, name="t")
    report = dedup_kept_for_video(
        project=project, video_stem="vid",
        cfg=DedupConfig(distance_threshold=0.05),
    )
    assert report.inspected == 0
    assert report.removed == 0


def test_thresholds_dedup_round_trips_through_json(tmp_path: Path):
    """The dedup section must survive ``Thresholds.to_json`` →
    ``from_json`` so a project's saved overrides don't lose the
    threshold tweaks after a restart."""
    t = Thresholds()
    t.dedup.distance_threshold = 0.07
    t.dedup.lookback_frames = 500
    t.dedup.move_to_rejected = False

    path = tmp_path / "t.json"
    t.to_json(path)
    loaded = Thresholds.from_json(path)
    assert loaded.dedup.distance_threshold == 0.07
    assert loaded.dedup.lookback_frames == 500
    assert loaded.dedup.move_to_rejected is False


def test_thresholds_from_json_tolerates_missing_dedup_section(tmp_path: Path):
    """Older projects' threshold files don't have a dedup section — load with
    defaults rather than crashing."""
    path = tmp_path / "t.json"
    path.write_text(json.dumps({
        "scene": {"threshold": 27.0, "min_scene_len_frames": 8},
    }))
    loaded = Thresholds.from_json(path)
    assert loaded.dedup.distance_threshold == 0.02
    assert loaded.dedup.lookback_frames == 1000
    assert loaded.dedup.move_to_rejected is True


def test_thresholds_from_json_tolerates_legacy_enabled_field(tmp_path: Path):
    """A project saved before dedup became always-on may have
    ``dedup.enabled`` in its persisted JSON — must load without
    crashing on the unknown kwarg. The field is silently dropped."""
    path = tmp_path / "t.json"
    path.write_text(json.dumps({
        "dedup": {
            "enabled": False,           # legacy key; should be ignored
            "distance_threshold": 0.07,
            "move_to_rejected": False,
        },
    }))
    loaded = Thresholds.from_json(path)
    assert loaded.dedup.distance_threshold == 0.07
    assert loaded.dedup.move_to_rejected is False
    # ``enabled`` field is gone from the dataclass entirely.
    assert not hasattr(loaded.dedup, "enabled")


def test_dedup_metadata_appends_kept_false_record_for_drops(tmp_path: Path):
    """After dedup, the metadata log must contain a kept=False row per dropped
    frame so the frames API's last-write-wins logic flips them to rejected
    in the UI without needing a full rerun."""
    project_root = tmp_path / "proj"
    project = Project.create(project_root, name="t")

    # Two image files.
    a = project.kept_dir / "vid__s000_t000_f000000.png"
    b = project.kept_dir / "vid__s000_t000_f000001.png"
    Image.new("RGB", (16, 16), (10, 10, 10)).save(a)
    Image.new("RGB", (16, 16), (10, 10, 10)).save(b)

    # Seed metadata so the dedup append uses real per-frame data.
    log = MetadataLog(project.metadata_path)
    for png, score in ((a, 0.9), (b, 0.5)):
        log.append(FrameRecord(
            filename=png.stem, kept=True, scene_idx=0, tracklet_id=0,
            frame_idx=0, timestamp_seconds=0.0, bbox=(0, 0, 16, 16),
            ccip_distance=0.0, sharpness=1.0, visibility=1.0, aspect=1.0,
            score=score, video_stem="vid",
        ))

    # Stand-in for the real CCIP path: simulate "b is a dup of a" by directly
    # invoking the metadata-append helper with our chosen drop set.
    from neme_anima.dedup import _append_dedup_metadata
    _append_dedup_metadata(project, "vid", [a, b], drop_indices={1})

    rows = list(MetadataLog(project.metadata_path).iter_records(video_stem="vid"))
    # Original two appends + one dedup append for b.
    assert len(rows) == 3
    last_for_b = next(r for r in rows[::-1] if r.filename == b.stem)
    assert last_for_b.kept is False
    # The original score is preserved for traceability.
    assert last_for_b.score == 0.5


def test_dedup_metadata_preserves_owning_character_slug(tmp_path: Path):
    """Regression: the kept=False row written by dedup must inherit the
    owning character's slug from the most-recent kept=True row.
    Previously the demotion silently relabelled every dedup-rejected
    frame as the project's default character, mis-attributing rejected-
    by-dedup frames in per-character listings (a kiyotaka frame demoted
    by dedup ended up listed under "default" in the rejected drawer)."""
    project_root = tmp_path / "proj"
    project = Project.create(project_root, name="t")

    a = project.kept_dir / "vid__s000_t000_f000000.png"
    Image.new("RGB", (16, 16), (10, 10, 10)).save(a)

    log = MetadataLog(project.metadata_path)
    log.append(FrameRecord(
        filename=a.stem, kept=True, scene_idx=0, tracklet_id=0,
        frame_idx=0, timestamp_seconds=0.0, bbox=(0, 0, 16, 16),
        ccip_distance=0.0, sharpness=1.0, visibility=1.0, aspect=1.0,
        score=0.9, video_stem="vid", character_slug="kiyotaka",
    ))

    from neme_anima.dedup import _append_dedup_metadata
    _append_dedup_metadata(project, "vid", [a], drop_indices={0})

    rows = list(MetadataLog(project.metadata_path).iter_records(video_stem="vid"))
    last = rows[-1]
    assert last.kept is False
    assert last.character_slug == "kiyotaka"


def test_create_project_helper(tmp_path: Path):
    """Sanity: the test fixture above relies on Project.create laying the
    expected directory skeleton — fail loud here if that contract drifts."""
    project_root = tmp_path / "p"
    p = Project.create(project_root, name="x")
    assert p.kept_dir.exists()
    assert p.rejected_dir.exists()
    assert p.metadata_path.parent.exists()
    # created_at is timezone-aware; the contract used by load() depends on it.
    assert p.created_at.tzinfo is not None
