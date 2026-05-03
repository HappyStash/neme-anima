"""Tests for the core-tags module: compute, prune, and per-frame lookup.

The compute path is integration-style (it reads on-disk sidecars) but
doesn't require GPU/CCIP — we seed kept frames via Project.create + raw
file writes. Pure-function helpers (``prune_tags``, ``prune_sidecar_text``)
are exercised independently.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from neme_anima.core_tags import (
    DEFAULT_BLACKLIST,
    compute_core_tags,
    core_tags_for_filename,
    prune_sidecar_text,
    prune_tags,
)
from neme_anima.storage.metadata import FrameRecord, MetadataLog
from neme_anima.storage.project import DEFAULT_CHARACTER_SLUG, Project


def _seed_kept_frame(
    project: Project, *, filename: str, character_slug: str, tags: str,
    description: str = "",
) -> None:
    """Write a real PNG + sidecar AND append a metadata row.

    Several core_tags tests need the metadata-log routing AND the on-disk
    sidecar in sync — exactly what the extraction pipeline produces — so
    this helper does both. The image content is irrelevant for these
    tests; we just need a present file so the sidecar reads succeed.
    """
    png = project.kept_dir / f"{filename}.png"
    Image.new("RGB", (8, 8), (5, 5, 5)).save(png)
    sidecar_text = f"{tags}\n"
    if description:
        sidecar_text += f"{description}\n"
    png.with_suffix(".txt").write_text(sidecar_text, encoding="utf-8")
    MetadataLog(project.metadata_path).append(FrameRecord(
        filename=filename, kept=True,
        scene_idx=0, tracklet_id=0, frame_idx=0,
        timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
        ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
        score=0.9, video_stem="ep01", character_slug=character_slug,
    ))


def test_prune_tags_drops_named_tags_preserving_order():
    """The order of remaining tags must be preserved — kohya weights
    earlier tags more, so reordering the line silently changes training."""
    out = prune_tags(
        "1girl, blue_eyes, brown_hair, school_uniform, smile",
        ["blue_eyes", "brown_hair"],
    )
    assert out == "1girl, school_uniform, smile"


def test_prune_tags_no_match_returns_input():
    out = prune_tags("1girl, blue_eyes", ["red_hair"])
    assert out == "1girl, blue_eyes"


def test_prune_tags_empty_core_set_is_noop():
    """Disabling pruning short-circuits — the tag line is byte-identical."""
    line = "1girl, blue_eyes, brown_hair"
    assert prune_tags(line, []) == line


def test_prune_sidecar_preserves_description():
    """Pruning operates on line 1 only; line 2 (LLM description) survives
    untouched. This is the contract the trainer reads — losing the
    description would silently degrade caption quality."""
    text = "1girl, blue_eyes, smile\nA young woman with a determined look.\n"
    out = prune_sidecar_text(text, ["blue_eyes"])
    assert out == "1girl, smile\nA young woman with a determined look.\n"


def test_prune_sidecar_no_description_stays_one_line():
    """Round-tripping a one-line sidecar must stay one-line — adding a
    phantom newline would diff every existing project on first staging."""
    out = prune_sidecar_text("1girl, blue_eyes\n", ["blue_eyes"])
    assert out == "1girl\n"


def test_compute_core_tags_empty_corpus(tmp_path: Path):
    """A character with zero kept frames returns an empty report — no
    division-by-zero, and the UI can render the empty state cleanly."""
    project = Project.create(tmp_path / "p", name="show")
    report = compute_core_tags(
        project=project, character_slug=DEFAULT_CHARACTER_SLUG,
    )
    assert report.corpus_size == 0
    assert report.tags == ()


def test_compute_core_tags_returns_frequent_tags(tmp_path: Path):
    """Tags appearing in ≥ 35 % of a character's frames become core tags
    (sorted desc by frequency). Below-threshold tags are dropped silently
    — the report only carries what would actually be pruned."""
    project = Project.create(tmp_path / "p", name="show")
    # Seed 4 frames; "blue_eyes" appears in all 4 (100 %), "brown_hair" in
    # 3 (75 %), "school_uniform" in 1 (25 %, below threshold).
    _seed_kept_frame(project, filename="ep01__a",
                     character_slug=DEFAULT_CHARACTER_SLUG,
                     tags="1girl, blue_eyes, brown_hair, school_uniform")
    _seed_kept_frame(project, filename="ep01__b",
                     character_slug=DEFAULT_CHARACTER_SLUG,
                     tags="1girl, blue_eyes, brown_hair")
    _seed_kept_frame(project, filename="ep01__c",
                     character_slug=DEFAULT_CHARACTER_SLUG,
                     tags="1girl, blue_eyes, brown_hair")
    _seed_kept_frame(project, filename="ep01__d",
                     character_slug=DEFAULT_CHARACTER_SLUG,
                     tags="1girl, blue_eyes")
    report = compute_core_tags(
        project=project, character_slug=DEFAULT_CHARACTER_SLUG,
        threshold=0.35,
    )
    tag_names = [t for t, _ in report.tags]
    # "1girl" is in DEFAULT_BLACKLIST (pose/composition family) so it gets
    # filtered out even at 100 %; "blue_eyes" + "brown_hair" survive.
    assert "blue_eyes" in tag_names
    assert "brown_hair" in tag_names
    assert "school_uniform" not in tag_names  # below threshold
    assert "1girl" not in tag_names  # blacklisted
    assert "1girl" in report.blacklisted


def test_compute_core_tags_respects_threshold(tmp_path: Path):
    """Lowering the threshold pulls in less-frequent tags; raising it
    eliminates more. Same corpus, two thresholds, distinct outcomes."""
    project = Project.create(tmp_path / "p", name="show")
    for i in range(4):
        _seed_kept_frame(
            project, filename=f"ep01__{i}",
            character_slug=DEFAULT_CHARACTER_SLUG,
            tags=("blue_eyes, hairband"
                  if i < 1 else "blue_eyes"),  # hairband at 25 %
        )
    high = compute_core_tags(project=project, character_slug=DEFAULT_CHARACTER_SLUG, threshold=0.5)
    low = compute_core_tags(project=project, character_slug=DEFAULT_CHARACTER_SLUG, threshold=0.2)
    assert "hairband" not in [t for t, _ in high.tags]
    assert "hairband" in [t for t, _ in low.tags]


def test_compute_core_tags_filters_to_specified_character(tmp_path: Path):
    """A frame routed to character X never contributes to character Y's
    core tag set, even if both share a tag — the per-character corpus is
    metadata-driven, not file-system driven."""
    project = Project.create(tmp_path / "p", name="show")
    project.add_character(name="Mio")
    _seed_kept_frame(project, filename="ep01__yui_a",
                     character_slug=DEFAULT_CHARACTER_SLUG,
                     tags="brown_hair, hairband")
    _seed_kept_frame(project, filename="ep01__yui_b",
                     character_slug=DEFAULT_CHARACTER_SLUG,
                     tags="brown_hair, hairband")
    _seed_kept_frame(project, filename="ep01__mio_a",
                     character_slug="mio",
                     tags="long_hair, black_hair")
    yui = compute_core_tags(
        project=project, character_slug=DEFAULT_CHARACTER_SLUG,
    )
    mio = compute_core_tags(project=project, character_slug="mio")
    assert {t for t, _ in yui.tags} == {"brown_hair", "hairband"}
    assert {t for t, _ in mio.tags} == {"long_hair", "black_hair"}


def test_compute_core_tags_custom_blacklist_override(tmp_path: Path):
    """Callers can override the default blacklist — useful for e.g. anime
    LoRAs targeting a stylized franchise where 'open_mouth' really IS a
    character signature."""
    project = Project.create(tmp_path / "p", name="show")
    _seed_kept_frame(project, filename="ep01__a",
                     character_slug=DEFAULT_CHARACTER_SLUG,
                     tags="blue_eyes, open_mouth")
    # Default blacklist eats open_mouth.
    default_report = compute_core_tags(
        project=project, character_slug=DEFAULT_CHARACTER_SLUG,
    )
    assert "open_mouth" not in [t for t, _ in default_report.tags]
    # Empty blacklist preserves it.
    permissive = compute_core_tags(
        project=project, character_slug=DEFAULT_CHARACTER_SLUG,
        blacklist=(),
    )
    assert "open_mouth" in [t for t, _ in permissive.tags]


def test_default_blacklist_includes_known_pose_tags():
    """Sanity: the shipped blacklist covers the obvious pose/composition
    tags so a fresh project gets reasonable results out of the box."""
    for tag in ("solo", "1girl", "looking_at_viewer", "standing", "smile"):
        assert tag in DEFAULT_BLACKLIST


def test_core_tags_for_filename_returns_owner_and_list(tmp_path: Path):
    """When a frame's owning character has core_tags_enabled, the helper
    returns its persisted list — the staging step uses this to prune."""
    project = Project.create(tmp_path / "p", name="show")
    project.characters[0].core_tags = ["blue_eyes", "hairband"]
    project.characters[0].core_tags_enabled = True
    project.save()
    _seed_kept_frame(project, filename="ep01__a",
                     character_slug=DEFAULT_CHARACTER_SLUG,
                     tags="blue_eyes, hairband, smile")
    out = core_tags_for_filename(Project.load(project.root), "ep01__a")
    assert out == (DEFAULT_CHARACTER_SLUG, ["blue_eyes", "hairband"])


def test_core_tags_for_filename_disabled_returns_empty_list(tmp_path: Path):
    """When pruning is disabled for the owning character, callers see an
    empty core_tags list — but the slug is still returned so the staging
    step can tag the frame correctly even without pruning."""
    project = Project.create(tmp_path / "p", name="show")
    project.characters[0].core_tags = ["blue_eyes"]
    project.characters[0].core_tags_enabled = False
    project.save()
    _seed_kept_frame(project, filename="ep01__a",
                     character_slug=DEFAULT_CHARACTER_SLUG, tags="blue_eyes")
    out = core_tags_for_filename(Project.load(project.root), "ep01__a")
    assert out == (DEFAULT_CHARACTER_SLUG, [])


def test_core_tags_for_filename_unknown_filename_returns_none(tmp_path: Path):
    """A filename that has no metadata row should return None so the
    caller treats the frame as un-prunable rather than crashing."""
    project = Project.create(tmp_path / "p", name="show")
    out = core_tags_for_filename(project, "ghost_frame")
    assert out is None


def test_compute_core_tags_ignores_deleted_frames(tmp_path: Path):
    """Frames whose .png has been removed from disk (e.g. via the Frames
    tab's delete) must NOT count toward ``corpus_size``. The metadata log
    is append-only, so a stale row for a deleted frame would otherwise
    inflate the denominator and crush every surviving tag below the
    threshold (the bug the user hit: 15 curated frames showing
    corpus=275)."""
    project = Project.create(tmp_path / "p", name="show")
    # Seed 3 surviving frames with "blue_eyes" in all of them (100 %).
    for i in range(3):
        _seed_kept_frame(
            project, filename=f"ep01__keep_{i}",
            character_slug=DEFAULT_CHARACTER_SLUG,
            tags="blue_eyes, brown_hair",
        )
    # Seed 17 metadata rows for frames whose .png has been deleted from
    # disk: the metadata stays but the file is gone. Without the on-disk
    # filter, corpus_size would be 20 → "blue_eyes" frequency = 3/20 = 15 %
    # → below the default 35 % threshold → empty suggestions.
    for i in range(17):
        MetadataLog(project.metadata_path).append(FrameRecord(
            filename=f"ep01__deleted_{i}", kept=True,
            scene_idx=0, tracklet_id=0, frame_idx=0,
            timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
            ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
            score=0.9, video_stem="ep01", character_slug=DEFAULT_CHARACTER_SLUG,
        ))

    report = compute_core_tags(
        project=project, character_slug=DEFAULT_CHARACTER_SLUG,
        threshold=0.35,
    )
    assert report.corpus_size == 3, report.corpus_size
    tag_names = [t for t, _ in report.tags]
    assert "blue_eyes" in tag_names
    assert "brown_hair" in tag_names


def test_compute_core_tags_excludes_crop_derivatives(tmp_path: Path):
    """Crop derivatives (``<original>_crop.png``) get their own metadata
    row but share the original's .txt sidecar. Counting them in
    ``corpus_size`` would inflate the denominator with zero-tag
    contribution and crush every surviving tag below the threshold.
    Mirror the deleted-frame fix: exclude ``_crop`` rows."""
    from neme_anima.storage.metadata import FrameRecord
    from neme_anima.storage.project import CROP_SUFFIX

    project = Project.create(tmp_path / "p", name="show")
    # 4 originals, all with "blue_eyes".
    for i in range(4):
        _seed_kept_frame(
            project, filename=f"ep01__a_{i}",
            character_slug=DEFAULT_CHARACTER_SLUG,
            tags="blue_eyes",
        )
    # 4 crop derivatives — png on disk + metadata row, but no separate
    # sidecar. They must NOT count toward corpus_size.
    for i in range(4):
        crop_name = f"ep01__a_{i}{CROP_SUFFIX}"
        Image.new("RGB", (8, 8), (5, 5, 5)).save(
            project.kept_dir / f"{crop_name}.png",
        )
        MetadataLog(project.metadata_path).append(FrameRecord(
            filename=crop_name, kept=True,
            scene_idx=0, tracklet_id=0, frame_idx=0,
            timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
            ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
            score=0.9, video_stem="ep01", character_slug=DEFAULT_CHARACTER_SLUG,
        ))

    report = compute_core_tags(
        project=project, character_slug=DEFAULT_CHARACTER_SLUG,
        threshold=0.35,
    )
    assert report.corpus_size == 4, report.corpus_size
    tag_names = [t for t, _ in report.tags]
    assert "blue_eyes" in tag_names
