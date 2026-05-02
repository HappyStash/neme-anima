"""Tests for the multi-character data model + auto-migration.

Covers project.json migration from the legacy single-character shape, the
``characters`` round-trip through save/load, the per-character API on
``add_ref`` / ``effective_refs_for`` / ``set_excluded_refs``, and the dict
shape of ``Source.excluded_refs`` with auto-migration from old list shape.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from neme_anima.storage.project import (
    DEFAULT_CHARACTER_SLUG,
    Character,
    Project,
    _slugify_character_name,
)


def _legacy_project_json(
    *, root: Path, name: str, refs: list[dict], sources: list[dict],
    training: dict | None = None,
) -> Path:
    """Write a project.json in the old single-character shape (no characters
    field, top-level refs + training, list-shape excluded_refs)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "refs").mkdir(exist_ok=True)
    (root / "output" / "kept").mkdir(parents=True, exist_ok=True)
    out = {
        "name": name,
        "slug": root.name,
        "created_at": datetime.now(UTC).isoformat(),
        "sources": sources,
        "refs": refs,
        "thresholds_overrides": {},
        "source_root": None,
        "pause_before_tag": True,
        "llm": {"enabled": False, "endpoint": "http://localhost:1234",
                "model": "", "prompt": "", "api_key": ""},
        "training": training or {},
    }
    (root / "project.json").write_text(json.dumps(out, indent=2))
    return root / "project.json"


def test_create_seeds_one_default_character(tmp_path: Path):
    """A fresh project always lands with exactly one default character —
    every character-aware code path can rely on ``project.characters[0]``
    existing the moment a project is loaded."""
    p = Project.create(tmp_path / "p", name="Megumin")
    assert len(p.characters) == 1
    c = p.characters[0]
    assert c.slug == DEFAULT_CHARACTER_SLUG
    assert c.name == "Megumin"
    assert c.refs == []
    assert c.trigger_token == ""


def test_load_legacy_project_synthesizes_default_character(tmp_path: Path):
    """A pre-multi-character project.json (top-level refs/training, no
    ``characters`` array) loads into a single default character carrying the
    legacy refs and training so existing projects open unchanged."""
    root = tmp_path / "legacy"
    _legacy_project_json(
        root=root, name="Yui",
        refs=[{"path": "/abs/path/yui_face.png",
               "added_at": "2024-01-01T00:00:00+00:00"}],
        sources=[],
        training={"preset": "character", "rank": 64},
    )
    p = Project.load(root)
    assert len(p.characters) == 1
    c = p.characters[0]
    assert c.slug == DEFAULT_CHARACTER_SLUG
    assert c.name == "Yui"
    assert len(c.refs) == 1
    assert c.refs[0].path == "/abs/path/yui_face.png"
    # Training fields survive the migration so the user's training settings
    # don't reset to defaults on first open under the new code.
    assert c.training.preset == "character"
    assert c.training.rank == 64


def test_load_legacy_excluded_refs_list_migrates_to_dict(tmp_path: Path):
    """Old projects stored ``excluded_refs`` as a flat list — those land
    under the default character's slug, so per-character opt-outs work
    immediately for any later character that gets added."""
    root = tmp_path / "legacy"
    _legacy_project_json(
        root=root, name="legacy",
        refs=[],
        sources=[{
            "path": "/abs/ep01.mkv",
            "added_at": "2024-01-01T00:00:00+00:00",
            "excluded_refs": ["/abs/refs/r1.png", "/abs/refs/r2.png"],
        }],
    )
    p = Project.load(root)
    assert p.sources[0].excluded_refs == {
        DEFAULT_CHARACTER_SLUG: ["/abs/refs/r1.png", "/abs/refs/r2.png"],
    }


def test_load_legacy_empty_excluded_refs_normalizes_to_empty_dict(tmp_path: Path):
    """An empty list shouldn't leave a stub ``{"default": []}`` behind —
    the in-memory shape stays an empty dict so saves don't write empty
    keys back to disk."""
    root = tmp_path / "legacy"
    _legacy_project_json(
        root=root, name="legacy",
        refs=[],
        sources=[{
            "path": "/abs/ep01.mkv",
            "added_at": "2024-01-01T00:00:00+00:00",
            "excluded_refs": [],
        }],
    )
    p = Project.load(root)
    assert p.sources[0].excluded_refs == {}


def test_save_round_trips_characters(tmp_path: Path):
    """Adding a second character and persisting the project — the round-trip
    must preserve slug, name, refs, trigger token, and training overrides."""
    p = Project.create(tmp_path / "p", name="K-On!")
    c2 = p.add_character(name="Mio")
    c2.trigger_token = "mio_kon"
    c2.training.preset = "character"
    c2.training.rank = 64
    p.save()

    reloaded = Project.load(p.root)
    assert [c.slug for c in reloaded.characters] == [DEFAULT_CHARACTER_SLUG, "mio"]
    assert reloaded.characters[1].name == "Mio"
    assert reloaded.characters[1].trigger_token == "mio_kon"
    assert reloaded.characters[1].training.preset == "character"
    assert reloaded.characters[1].training.rank == 64


def test_add_character_uniquifies_colliding_slugs(tmp_path: Path):
    """Two characters with the same name don't share a slug — slugs are the
    primary key and must be globally unique within a project."""
    p = Project.create(tmp_path / "p", name="show")
    p.add_character(name="Yui")
    p.add_character(name="Yui")
    p.add_character(name="Yui")
    slugs = [c.slug for c in p.characters]
    # First was the auto-created default; the three Yuis come after.
    assert slugs == [DEFAULT_CHARACTER_SLUG, "yui", "yui-2", "yui-3"]


def test_add_character_slug_param_takes_precedence_over_name(tmp_path: Path):
    """Explicit slug overrides the name-derived one — useful when the
    display name has unicode/diacritics the user doesn't want in paths."""
    p = Project.create(tmp_path / "p", name="show")
    c = p.add_character(name="ユイ", slug="yui")
    assert c.slug == "yui"


def test_remove_character_strips_per_source_optouts(tmp_path: Path):
    """Removing a character also drops its key from every source's
    excluded_refs map, so dangling slug keys don't accumulate."""
    p = Project.create(tmp_path / "p", name="show")
    p.add_character(name="Mio")
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    img = tmp_path / "r.png"; img.write_bytes(b"X")
    r = p.add_ref(img, character_slug="mio")
    p.add_source(vid)
    p.set_excluded_refs(0, [r.path], character_slug="mio")
    assert "mio" in p.sources[0].excluded_refs
    p.remove_character("mio")
    assert "mio" not in p.sources[0].excluded_refs


def test_remove_character_refuses_to_drop_last_one(tmp_path: Path):
    """A project with no characters has no useful state — refuse rather
    than silently re-creating one on next save."""
    p = Project.create(tmp_path / "p", name="show")
    with pytest.raises(ValueError, match="last character"):
        p.remove_character(DEFAULT_CHARACTER_SLUG)


def test_add_ref_targets_specified_character(tmp_path: Path):
    """add_ref(character_slug=...) puts the ref on the named character only,
    not on the default. Default character stays empty in this case."""
    p = Project.create(tmp_path / "p", name="show")
    p.add_character(name="Mio")
    img = tmp_path / "mio.png"; img.write_bytes(b"X")
    r = p.add_ref(img, character_slug="mio")
    assert p.character_by_slug("mio").refs == [r]
    assert p.characters[0].refs == []  # default character untouched


def test_remove_ref_strips_from_every_character(tmp_path: Path):
    """If a ref were ever shared (we don't do this today, but the model
    permits it via filesystem identity), deleting the path strips it from
    every character that referenced it. The on-disk file is then unlinked
    only after no references remain — guards against shared-path bugs."""
    p = Project.create(tmp_path / "p", name="show")
    p.add_character(name="Mio")
    img = tmp_path / "ref.png"; img.write_bytes(b"X")
    r = p.add_ref(img, character_slug=DEFAULT_CHARACTER_SLUG)
    # Manually duplicate the entry into Mio's refs to simulate sharing.
    p.character_by_slug("mio").refs.append(r)
    p.save()
    p.remove_ref(r.path)
    assert p.characters[0].refs == []
    assert p.character_by_slug("mio").refs == []


def test_effective_refs_per_character(tmp_path: Path):
    """effective_refs_for(source_idx, character_slug=...) returns each
    character's refs minus that character's per-source opt-outs — the
    other character's refs and opt-outs are invisible."""
    p = Project.create(tmp_path / "p", name="show")
    p.add_character(name="Mio")
    img_yui = tmp_path / "y.png"; img_yui.write_bytes(b"Y")
    img_mio = tmp_path / "m.png"; img_mio.write_bytes(b"M")
    yr = p.add_ref(img_yui, character_slug=DEFAULT_CHARACTER_SLUG)
    mr = p.add_ref(img_mio, character_slug="mio")
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    p.add_source(vid)
    # Opt out Mio's only ref for this source.
    p.set_excluded_refs(0, [mr.path], character_slug="mio")
    assert p.effective_refs_for(0) == [yr.path]
    assert p.effective_refs_for(0, character_slug="mio") == []


def test_effective_refs_unknown_slug_raises(tmp_path: Path):
    """An unknown character_slug is a caller bug — raise instead of silently
    rerouting to the default character, which would mask UI/state issues."""
    p = Project.create(tmp_path / "p", name="show")
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    p.add_source(vid)
    with pytest.raises(KeyError, match="unknown character slug"):
        p.effective_refs_for(0, character_slug="nope")


def test_project_refs_property_aliases_default_character(tmp_path: Path):
    """The legacy ``project.refs`` accessor keeps reading the default
    character — this is the bridge that lets existing API endpoints + the
    mono-character UI keep working unchanged."""
    p = Project.create(tmp_path / "p", name="show")
    img = tmp_path / "r.png"; img.write_bytes(b"X")
    p.add_ref(img)
    assert len(p.refs) == 1
    assert p.refs == p.characters[0].refs


def test_project_training_property_aliases_default_character(tmp_path: Path):
    """The legacy ``project.training`` accessor reads + writes the default
    character's training config; the existing TrainingConfig UI flows
    against it untouched."""
    p = Project.create(tmp_path / "p", name="show")
    p.training.rank = 128
    assert p.characters[0].training.rank == 128


def test_slugify_character_name_handles_unicode_and_punctuation():
    """Slugs are filesystem-safe ASCII; unicode + punctuation collapse to
    single hyphens, leading/trailing hyphens are stripped."""
    assert _slugify_character_name("Yui") == "yui"
    assert _slugify_character_name("Mio Akiyama") == "mio-akiyama"
    assert _slugify_character_name("---weird name---") == "weird-name"
    assert _slugify_character_name("") == "character"
    assert _slugify_character_name("ユイ") == "character"


def test_character_dataclass_default_training_is_independent_per_instance():
    """Two characters' training configs must not share a single mutable
    instance — the dataclass uses default_factory, but a regression here
    would silently couple every character's config."""
    a = Character(slug="a", name="A")
    b = Character(slug="b", name="B")
    a.training.rank = 99
    assert b.training.rank != 99
