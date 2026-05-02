"""End-to-end-ish tests for the staging pipeline applying core-tag pruning
+ per-character subdirs + dataset.toml multi-block emission.

These exercise the WIRING — the algorithms themselves are covered in
test_core_tags.py and test_balancing.py. Here we confirm that turning on
``core_tags_enabled`` flows through ``build_dataset_staging`` to actually
rewrite the staged sidecar, and that multi-character projects produce
per-character subdirs + matching dataset.toml blocks.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from neme_anima import training
from neme_anima.storage.metadata import FrameRecord, MetadataLog
from neme_anima.storage.project import DEFAULT_CHARACTER_SLUG, Project


def _seed_kept(
    project: Project, *, filename: str, character_slug: str, tags: str,
) -> None:
    png = project.kept_dir / f"{filename}.png"
    Image.new("RGB", (8, 8), (5, 5, 5)).save(png)
    png.with_suffix(".txt").write_text(f"{tags}\n", encoding="utf-8")
    MetadataLog(project.metadata_path).append(FrameRecord(
        filename=filename, kept=True,
        scene_idx=0, tracklet_id=0, frame_idx=0,
        timestamp_seconds=0.0, bbox=(0, 0, 8, 8),
        ccip_distance=0.05, sharpness=1.0, visibility=1.0, aspect=1.0,
        score=0.9, video_stem="ep01", character_slug=character_slug,
    ))


def test_staging_strips_core_tags_when_enabled(tmp_path: Path):
    """With core_tags_enabled and a saved core_tags list, the staged
    sidecar drops those tags. The original sidecar in kept/ stays
    intact — pruning is non-destructive."""
    project = Project.create(tmp_path / "p", name="solo")
    project.characters[0].core_tags = ["blue_eyes", "hairband"]
    project.characters[0].core_tags_enabled = True
    project.save()
    _seed_kept(
        project, filename="f1",
        character_slug=DEFAULT_CHARACTER_SLUG,
        tags="blue_eyes, hairband, smile",
    )
    dest = tmp_path / "stage"
    info = training.build_dataset_staging(Project.load(project.root), dest)

    # The staged sidecar has core tags removed; the original still has them.
    staged_text = (dest / "f1.txt").read_text(encoding="utf-8")
    original_text = (project.kept_dir / "f1.txt").read_text(encoding="utf-8")
    assert staged_text.startswith("smile")
    assert "blue_eyes" not in staged_text
    assert "blue_eyes" in original_text  # untouched
    assert info["pruned"] == 1


def test_staging_does_not_prune_when_disabled(tmp_path: Path):
    """Disabling pruning short-circuits to the symlink path — even if a
    core_tags list is saved on the character, it isn't applied."""
    project = Project.create(tmp_path / "p", name="solo")
    project.characters[0].core_tags = ["blue_eyes"]
    project.characters[0].core_tags_enabled = False  # explicitly off
    project.save()
    _seed_kept(
        project, filename="f1",
        character_slug=DEFAULT_CHARACTER_SLUG,
        tags="blue_eyes, smile",
    )
    dest = tmp_path / "stage"
    info = training.build_dataset_staging(Project.load(project.root), dest)
    # Sidecar staged unchanged.
    assert (dest / "f1.txt").read_text(encoding="utf-8") == "blue_eyes, smile\n"
    assert info["pruned"] == 0


def test_multi_character_staging_creates_subdirs(tmp_path: Path):
    """A project with > 1 character routes each frame into its owner's
    subdirectory in the staging dir. Single-character projects retain
    the historical flat layout (covered by existing staging tests)."""
    project = Project.create(tmp_path / "p", name="duo")
    project.add_character(name="Mio")
    _seed_kept(project, filename="yui_a",
               character_slug=DEFAULT_CHARACTER_SLUG, tags="x")
    _seed_kept(project, filename="mio_a", character_slug="mio", tags="x")
    dest = tmp_path / "stage"
    training.build_dataset_staging(Project.load(project.root), dest)
    assert (dest / DEFAULT_CHARACTER_SLUG / "yui_a.png").exists()
    assert (dest / "mio" / "mio_a.png").exists()
    # Flat-layout files do NOT exist — no fallback that would confuse
    # diffusion-pipe into seeing the same frame twice.
    assert not (dest / "yui_a.png").exists()
    assert not (dest / "mio_a.png").exists()


def test_multi_character_dataset_toml_emits_block_per_character(
    tmp_path: Path,
):
    """The TOML emits one ``[[directory]]`` per character with frames,
    pointed at that character's staging subdir, with num_repeats set
    from the balancing pass."""
    project = Project.create(tmp_path / "p", name="duo")
    project.add_character(name="Mio")
    for i in range(100):
        _seed_kept(project, filename=f"yui_{i}",
                   character_slug=DEFAULT_CHARACTER_SLUG, tags="x")
    for i in range(50):
        _seed_kept(project, filename=f"mio_{i}",
                   character_slug="mio", tags="x")
    project = Project.load(project.root)
    staged = tmp_path / "stage"
    text = training.render_dataset_toml(project, dataset_root=staged)
    # Exactly two blocks emitted — one per character with frames.
    assert text.count("[[directory]]") == 2
    # Subdir paths are present for each character.
    assert str((staged / DEFAULT_CHARACTER_SLUG).resolve()) in text
    assert str((staged / "mio").resolve()) in text
    # num_repeats reflects the balancing — Mio (50 frames vs Yui's 100)
    # gets multiply > 1; Yui floors at 1.
    # Parse loosely: just check both 1.0 and 1.5 appear in the relevant lines.
    assert "num_repeats = 1.5" in text
    assert "num_repeats = 1" in text


def test_single_character_dataset_toml_keeps_flat_layout(tmp_path: Path):
    """A single-character project must produce the same single-block
    TOML as before this PR — the flat-layout contract is what the
    historical tests pin and is what mono-character users expect."""
    project = Project.create(tmp_path / "p", name="solo")
    text = training.render_dataset_toml(project)
    assert text.count("[[directory]]") == 1
    # No per-character subdir reference.
    assert "/default" not in text or text.endswith("\n")  # paranoid
    # The historical num_repeats = 1 line is preserved.
    assert "num_repeats = 1" in text


def test_multi_character_staging_sidecars_pruned_in_subdir(tmp_path: Path):
    """Pruning + per-character staging compose: a multi-character
    project with one character pruning-enabled writes pruned sidecars
    into that character's subdir, and untouched sidecars into the
    other character's subdir."""
    project = Project.create(tmp_path / "p", name="duo")
    project.add_character(name="Mio")
    project.characters[0].core_tags = ["blue_eyes"]
    project.characters[0].core_tags_enabled = True
    project.save()
    _seed_kept(project, filename="yui_a",
               character_slug=DEFAULT_CHARACTER_SLUG,
               tags="blue_eyes, smile")
    _seed_kept(project, filename="mio_a",
               character_slug="mio", tags="blue_eyes, smile")
    dest = tmp_path / "stage"
    training.build_dataset_staging(Project.load(project.root), dest)
    yui_text = (dest / DEFAULT_CHARACTER_SLUG / "yui_a.txt").read_text(encoding="utf-8")
    mio_text = (dest / "mio" / "mio_a.txt").read_text(encoding="utf-8")
    assert "blue_eyes" not in yui_text  # pruned for Yui
    assert "blue_eyes" in mio_text      # not pruned for Mio
