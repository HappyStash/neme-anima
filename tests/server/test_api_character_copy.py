"""API smoke tests for POST /characters/{slug}/copy-to."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from neme_anima.server.app import create_app


def _setup(tmp_path: Path):
    """Create two registered projects on the test app."""
    from neme_anima.storage.metadata import FrameRecord, MetadataLog
    from neme_anima.storage.project import Project

    app = create_app(state_dir=tmp_path / "state")
    client = TestClient(app)

    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    Project.create(src_dir, name="src")
    Project.create(dst_dir, name="dst")

    # Register both with the running app.
    client.post("/api/projects/register", json={"folder": str(src_dir)})
    client.post("/api/projects/register", json={"folder": str(dst_dir)})

    # Seed src with a character + frame.
    src = Project.load(src_dir)
    src.add_character(name="Sora", slug="sora")
    log = MetadataLog(src.metadata_path)
    (src.kept_dir / "ep01__a.png").write_bytes(b"\x89PNG")
    log.append(FrameRecord(
        filename="ep01__a", kept=True, scene_idx=0, tracklet_id=0, frame_idx=0,
        timestamp_seconds=0.0, bbox=(0, 0, 1, 1), ccip_distance=0.0,
        sharpness=0.0, visibility=0.0, aspect=1.0, score=0.0,
        video_stem="ep01", character_slug="sora",
    ))

    return client, src, Project.load(dst_dir)


def test_copy_to_endpoint_happy_path(tmp_path):
    client, src, dst = _setup(tmp_path)
    resp = client.post(
        f"/api/projects/{src.slug}/characters/sora/copy-to",
        json={"destination_slug": dst.slug, "dry_run": False},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["character_slug"] == "sora"
    assert body["dry_run"] is False
    assert "ep01__a" in body["frames_added"]


def test_copy_to_endpoint_returns_409_on_slug_collision(tmp_path):
    client, src, dst = _setup(tmp_path)
    # Pre-create a "sora" in dst → collision.
    from neme_anima.storage.project import Project
    dst_proj = Project.load(dst.root)
    dst_proj.add_character(name="Sora-existing", slug="sora")
    resp = client.post(
        f"/api/projects/{src.slug}/characters/sora/copy-to",
        json={"destination_slug": dst.slug},
    )
    assert resp.status_code == 409


def test_copy_to_endpoint_returns_404_for_missing_destination(tmp_path):
    client, src, dst = _setup(tmp_path)
    resp = client.post(
        f"/api/projects/{src.slug}/characters/sora/copy-to",
        json={"destination_slug": "no-such-dst"},
    )
    assert resp.status_code == 404


def test_copy_to_endpoint_dry_run(tmp_path):
    client, src, dst = _setup(tmp_path)
    resp = client.post(
        f"/api/projects/{src.slug}/characters/sora/copy-to",
        json={"destination_slug": dst.slug, "dry_run": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dry_run"] is True
    # Nothing actually written.
    from neme_anima.storage.project import Project
    dst_after = Project.load(dst.root)
    assert dst_after.character_by_slug("sora") is None
