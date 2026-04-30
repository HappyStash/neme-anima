"""Tests for /api/projects/{slug}/sources routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from neme_extractor.server.app import create_app
from neme_extractor.storage.project import Project


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "p", name="p")


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


async def test_add_source(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"
    vid.write_bytes(b"")
    resp = await client.post(
        f"/api/projects/{project.slug}/sources",
        json={"paths": [str(vid)]},
    )
    assert resp.status_code == 200
    reloaded = Project.load(project.root)
    assert len(reloaded.sources) == 1


async def test_add_source_skips_duplicates(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"
    vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    assert resp.status_code == 200
    body = resp.json()
    # Endpoint reports skipped, not error.
    assert "skipped" in body


async def test_remove_source(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"
    vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.delete(f"/api/projects/{project.slug}/sources/0")
    assert resp.status_code == 204
    reloaded = Project.load(project.root)
    assert reloaded.sources == []


async def test_patch_excluded_refs(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    img = tmp_path / "ref.png"; img.write_bytes(b"")
    project.add_ref(img)
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.patch(
        f"/api/projects/{project.slug}/sources/0",
        json={"excluded_refs": [str(img.resolve())]},
    )
    assert resp.status_code == 200
    reloaded = Project.load(project.root)
    assert reloaded.sources[0].excluded_refs == [str(img.resolve())]


async def test_add_source_accepts_file_uri(client, project: Project, tmp_path: Path):
    vid = tmp_path / "Show E01.mkv"
    vid.write_bytes(b"")
    uri = f"file://{vid.as_posix().replace(' ', '%20')}"
    resp = await client.post(
        f"/api/projects/{project.slug}/sources",
        json={"paths": [uri]},
    )
    assert resp.status_code == 200
    reloaded = Project.load(project.root)
    assert len(reloaded.sources) == 1
    assert Path(reloaded.sources[0].path) == vid.resolve()


async def test_add_source_skips_browser_vfs_sentinel(client, project: Project):
    # When the browser hides the path, the frontend may still send the vfs:// fallback;
    # the server should reject it cleanly rather than try to open a junk path.
    resp = await client.post(
        f"/api/projects/{project.slug}/sources",
        json={"paths": ["vfs://Classroom of the Elite - S03E11 [1080p].mkv"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["added"] == []
    assert "vfs://" in body["skipped"][0]


async def test_import_folder_adds_all_videos(client, project: Project, tmp_path: Path):
    folder = tmp_path / "season3"
    folder.mkdir()
    (folder / "ep01.mkv").write_bytes(b"")
    (folder / "ep02.mp4").write_bytes(b"")
    (folder / "notes.txt").write_text("ignore")
    resp = await client.post(
        f"/api/projects/{project.slug}/sources/import-folder",
        json={"folder": str(folder)},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["added"]) == 2
    assert body["source_root"] == str(folder.resolve())
    reloaded = Project.load(project.root)
    assert reloaded.source_root == str(folder.resolve())


async def test_import_folder_rejects_missing_dir(client, project: Project, tmp_path: Path):
    resp = await client.post(
        f"/api/projects/{project.slug}/sources/import-folder",
        json={"folder": str(tmp_path / "nope")},
    )
    assert resp.status_code == 400


async def test_reimport_brings_back_deleted_rows(client, project: Project, tmp_path: Path):
    folder = tmp_path / "vids"; folder.mkdir()
    (folder / "ep01.mkv").write_bytes(b"")
    (folder / "ep02.mkv").write_bytes(b"")
    await client.post(
        f"/api/projects/{project.slug}/sources/import-folder",
        json={"folder": str(folder)},
    )
    # Delete one row.
    await client.delete(f"/api/projects/{project.slug}/sources/0")
    # Reimport — the deleted row should come back.
    resp = await client.post(f"/api/projects/{project.slug}/sources/reimport")
    assert resp.status_code == 200
    reloaded = Project.load(project.root)
    assert len(reloaded.sources) == 2


async def test_reimport_400_when_no_source_root(client, project: Project):
    resp = await client.post(f"/api/projects/{project.slug}/sources/reimport")
    assert resp.status_code == 400


async def test_thumbnail_extracts_from_real_video(
    client, project: Project, tmp_path: Path,
):
    """End-to-end: ffmpeg generates a sample mp4, the endpoint produces a JPEG."""
    import shutil as _shutil
    import subprocess

    if _shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    vid = tmp_path / "ep.mp4"
    res = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=duration=4:size=160x120:rate=24",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vid)],
        capture_output=True, text=True,
    )
    if res.returncode != 0 or not vid.exists():
        pytest.skip(f"ffmpeg sample generation failed: {res.stderr}")

    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.get(f"/api/projects/{project.slug}/sources/0/thumbnail")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content.startswith(b"\xff\xd8")
    # Cached file lives where we expect.
    assert (project.root / ".thumbs" / "ep.jpg").is_file()


async def test_thumbnail_serves_cached_jpeg(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    # Pre-populate the thumbnail cache so we don't have to actually decode video.
    thumbs_dir = project.root / ".thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    (thumbs_dir / "ep01.jpg").write_bytes(b"\xff\xd8\xff\xe0FAKEJPEG")
    resp = await client.get(f"/api/projects/{project.slug}/sources/0/thumbnail")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content.startswith(b"\xff\xd8")


async def test_thumbnail_404_when_video_missing(client, project: Project, tmp_path: Path):
    vid = tmp_path / "gone.mkv"; vid.write_bytes(b"")
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    vid.unlink()
    resp = await client.get(f"/api/projects/{project.slug}/sources/0/thumbnail")
    assert resp.status_code == 404


async def test_thumbnail_404_when_idx_out_of_range(client, project: Project):
    resp = await client.get(f"/api/projects/{project.slug}/sources/99/thumbnail")
    assert resp.status_code == 404


async def test_extract_enqueues_job(client, project: Project, tmp_path: Path):
    vid = tmp_path / "ep01.mkv"; vid.write_bytes(b"")
    img = tmp_path / "ref.png"; img.write_bytes(b"")
    project.add_ref(img)
    await client.post(f"/api/projects/{project.slug}/sources", json={"paths": [str(vid)]})
    resp = await client.post(f"/api/projects/{project.slug}/sources/0/extract")
    assert resp.status_code == 202
    assert "job_id" in resp.json()
