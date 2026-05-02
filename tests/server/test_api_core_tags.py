"""Tests for the new character endpoints: core-tags compute + PATCH fields,
balancing preview, and the staged-pruning behaviour exposed by training.

The character config persistence path (PATCH → save → load round-trip) is
covered with the API; the algorithmic behaviour is in test_core_tags.py
and test_balancing.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from neme_anima.server.app import create_app
from neme_anima.storage.metadata import FrameRecord, MetadataLog
from neme_anima.storage.project import DEFAULT_CHARACTER_SLUG, Project


@pytest.fixture
def project(tmp_path: Path) -> Project:
    p = Project.create(tmp_path / "p", name="K-On!")
    p.add_character(name="Mio")
    return p


@pytest.fixture
def app(tmp_path: Path, project: Project):
    a = create_app(state_dir=tmp_path / "state")
    a.state.registry.register(project)
    return a


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


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


async def test_core_tags_compute_endpoint(client, project: Project):
    """POST /core-tags/compute returns the suggested tag table without
    persisting anything. The character's stored core_tags stay empty
    until a follow-up PATCH saves what the user reviewed."""
    for i in range(4):
        _seed_kept(
            project, filename=f"f_{i}",
            character_slug=DEFAULT_CHARACTER_SLUG,
            tags="blue_eyes, hairband" if i < 3 else "blue_eyes",
        )
    resp = await client.post(
        f"/api/projects/{project.slug}/characters/"
        f"{DEFAULT_CHARACTER_SLUG}/core-tags/compute",
        json={},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["corpus_size"] == 4
    tag_names = [row["tag"] for row in body["tags"]]
    assert "blue_eyes" in tag_names    # 100 % → above default 0.35 threshold
    assert "hairband" in tag_names     # 75 % → still above
    # Nothing was saved to the character.
    reloaded = Project.load(project.root)
    assert reloaded.character_by_slug(DEFAULT_CHARACTER_SLUG).core_tags == []


async def test_core_tags_compute_respects_threshold_override(
    client, project: Project,
):
    for i in range(10):
        _seed_kept(
            project, filename=f"f_{i}",
            character_slug=DEFAULT_CHARACTER_SLUG,
            tags="blue_eyes, hairband" if i < 4 else "blue_eyes",
        )
    # hairband at 4/10 = 40 %. Threshold 0.5 drops it (below cutoff);
    # threshold 0.4 keeps it (>= cutoff). Verify both directions so a
    # future change to the comparison operator is caught here.
    above = await client.post(
        f"/api/projects/{project.slug}/characters/"
        f"{DEFAULT_CHARACTER_SLUG}/core-tags/compute",
        json={"threshold": 0.5},
    )
    above_names = [r["tag"] for r in above.json()["tags"]]
    assert "blue_eyes" in above_names
    assert "hairband" not in above_names

    below = await client.post(
        f"/api/projects/{project.slug}/characters/"
        f"{DEFAULT_CHARACTER_SLUG}/core-tags/compute",
        json={"threshold": 0.4},
    )
    below_names = [r["tag"] for r in below.json()["tags"]]
    assert "hairband" in below_names


async def test_patch_character_persists_core_tags_fields(
    client, project: Project,
):
    """PATCH carries the user-confirmed core_tags + threshold + enabled
    flag; reload sees them. Empty/whitespace tags are stripped server-
    side so a trailing comma in the UI doesn't store a phantom tag."""
    resp = await client.patch(
        f"/api/projects/{project.slug}/characters/{DEFAULT_CHARACTER_SLUG}",
        json={
            "core_tags": ["blue_eyes", "  hairband  ", "", "  "],
            "core_tags_freq_threshold": 0.4,
            "core_tags_enabled": True,
            "multiply": 2.5,
        },
    )
    assert resp.status_code == 200
    reloaded = Project.load(project.root)
    c = reloaded.character_by_slug(DEFAULT_CHARACTER_SLUG)
    assert c.core_tags == ["blue_eyes", "hairband"]
    assert c.core_tags_freq_threshold == 0.4
    assert c.core_tags_enabled is True
    assert c.multiply == 2.5


async def test_patch_character_clamps_threshold(client, project: Project):
    """A slider misfire (e.g. 0.0 or 99.9) shouldn't poison the saved
    config — clamp to (0.01, 1.0]. The clamp happens server-side so the
    invariant survives even hand-crafted API calls."""
    resp = await client.patch(
        f"/api/projects/{project.slug}/characters/{DEFAULT_CHARACTER_SLUG}",
        json={"core_tags_freq_threshold": 0.0},
    )
    assert resp.status_code == 200
    reloaded = Project.load(project.root)
    assert reloaded.character_by_slug(DEFAULT_CHARACTER_SLUG).core_tags_freq_threshold == 0.01


async def test_balancing_preview_endpoint(client, project: Project):
    """GET /balancing/preview returns the per-character row table with
    auto, manual (override), and effective multipliers — enough for the
    UI to render the column comparison without further math."""
    for i in range(100):
        _seed_kept(project, filename=f"yui_{i}",
                   character_slug=DEFAULT_CHARACTER_SLUG, tags="x")
    for i in range(50):
        _seed_kept(project, filename=f"mio_{i}",
                   character_slug="mio", tags="x")
    resp = await client.get(f"/api/projects/{project.slug}/balancing/preview")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_frames"] == 150
    by_slug = {r["character_slug"]: r for r in body["rows"]}
    assert by_slug[DEFAULT_CHARACTER_SLUG]["frame_count"] == 100
    assert by_slug["mio"]["frame_count"] == 50
    # Mio has fewer frames than the mean (75) so it gets a multiplier > 1.
    assert by_slug["mio"]["effective_multiply"] > 1.0
    # Yui is above the mean → floored at 1.0.
    assert by_slug[DEFAULT_CHARACTER_SLUG]["effective_multiply"] == 1.0


async def test_balancing_preview_unknown_project_404(client):
    resp = await client.get("/api/projects/ghost/balancing/preview")
    assert resp.status_code == 404
