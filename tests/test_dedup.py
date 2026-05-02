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


def test_dedup_disabled_is_noop(tmp_path: Path):
    """Disabled config must touch nothing — short-circuit before any I/O."""
    project_root = tmp_path / "proj"
    project = Project.create(project_root, name="t")
    png = project.kept_dir / "vid__s000_t000_f000000.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(png)

    report = dedup_kept_for_video(
        project=project, video_stem="vid",
        cfg=DedupConfig(enabled=False, distance_threshold=0.05),
    )
    assert report.removed == 0
    assert report.inspected == 0
    assert png.exists()


def test_dedup_no_kept_frames_is_noop(tmp_path: Path):
    """Empty kept_dir for the stem returns a clean zero report — guard against
    empty-directory crashes inside CCIP batch extract."""
    project_root = tmp_path / "proj"
    project = Project.create(project_root, name="t")
    report = dedup_kept_for_video(
        project=project, video_stem="vid",
        cfg=DedupConfig(enabled=True, distance_threshold=0.05),
    )
    assert report.inspected == 0
    assert report.removed == 0


def test_thresholds_dedup_round_trips_through_json(tmp_path: Path):
    """New section must survive ``Thresholds.to_json`` → ``from_json`` so a
    project's saved overrides don't lose the dedup field after a restart."""
    t = Thresholds()
    t.dedup.enabled = True
    t.dedup.distance_threshold = 0.07
    t.dedup.move_to_rejected = False

    path = tmp_path / "t.json"
    t.to_json(path)
    loaded = Thresholds.from_json(path)
    assert loaded.dedup.enabled is True
    assert loaded.dedup.distance_threshold == 0.07
    assert loaded.dedup.move_to_rejected is False


def test_thresholds_from_json_tolerates_missing_dedup_section(tmp_path: Path):
    """Older projects' threshold files don't have a dedup section — load with
    defaults rather than crashing on the new field."""
    path = tmp_path / "t.json"
    path.write_text(json.dumps({
        "scene": {"threshold": 27.0, "min_scene_len_frames": 8},
    }))
    loaded = Thresholds.from_json(path)
    assert loaded.dedup.enabled is False
    assert loaded.dedup.distance_threshold == 0.05


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
