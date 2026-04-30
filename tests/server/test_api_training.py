"""Tests for /api/projects/{slug}/training/* routes (config + path checks).

The actual ``start`` endpoint launches a subprocess and is therefore not
exercised here — those tests would require a real diffusion-pipe install.
We only assert that ``start`` refuses to launch when paths are missing
(the validate_for_run gate).
"""

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


async def test_get_config_returns_defaults_and_problems(client, project: Project):
    resp = await client.get(f"/api/projects/{project.slug}/training/config")
    assert resp.status_code == 200
    body = resp.json()
    cfg = body["config"]
    assert cfg["preset"] == "style"
    assert cfg["keep_last_n_checkpoints"] == 0
    assert cfg["llm_adapter_lr"] == 0.0
    # Empty paths should produce path_check errors and surface in problems.
    pc = body["path_checks"]
    assert all(pc[k]["error"] for k in (
        "diffusion_pipe_dir", "dit_path", "vae_path", "llm_path",
    ))
    assert len(body["problems"]) >= 4


async def test_patch_config_persists(client, project: Project, tmp_path: Path):
    resp = await client.patch(
        f"/api/projects/{project.slug}/training/config",
        json={
            "preset": "character",
            "learning_rate": 5e-5,
            "keep_last_n_checkpoints": 3,
            "trigger_token": "mychar",
            "resolutions": [768, 1024],
        },
    )
    assert resp.status_code == 200, resp.text
    cfg = resp.json()["config"]
    assert cfg["preset"] == "character"
    assert cfg["learning_rate"] == 5e-5
    assert cfg["keep_last_n_checkpoints"] == 3
    assert cfg["trigger_token"] == "mychar"
    assert cfg["resolutions"] == [768, 1024]
    # Re-read from disk to confirm persistence.
    reloaded = Project.load(project.root)
    assert reloaded.training.preset == "character"
    assert reloaded.training.learning_rate == 5e-5


async def test_check_path_missing(client, project: Project, tmp_path: Path):
    resp = await client.post(
        f"/api/projects/{project.slug}/training/check-path",
        json={"path": str(tmp_path / "nope"), "expect": "file"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert not body["exists"]
    assert "no such" in body["error"].lower()


async def test_check_path_existing_file(client, project: Project, tmp_path: Path):
    f = tmp_path / "some.bin"
    f.write_bytes(b"x")
    resp = await client.post(
        f"/api/projects/{project.slug}/training/check-path",
        json={"path": str(f), "expect": "file"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["exists"]
    assert body["is_file"]
    assert body["error"] is None


async def test_status_when_no_run(client, project: Project):
    resp = await client.get(f"/api/projects/{project.slug}/training/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["running"] is False
    assert body["state"] is None
    assert body["log_lines"] == []


async def test_start_refused_when_paths_missing(client, project: Project):
    # No paths set => validate_for_run returns problems => 409.
    resp = await client.post(f"/api/projects/{project.slug}/training/start")
    assert resp.status_code == 409
    assert "diffusion-pipe" in resp.json()["detail"]


async def test_resume_404s_when_no_runs(client, project: Project):
    resp = await client.post(f"/api/projects/{project.slug}/training/resume")
    assert resp.status_code == 409
    assert "no prior run" in resp.json()["detail"]


async def test_dataset_preview_shape(client, project: Project):
    resp = await client.get(
        f"/api/projects/{project.slug}/training/dataset-preview",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_images"] == 0
    assert body["samples"] == []


async def test_run_toml_preview_renders(client, project: Project):
    resp = await client.get(
        f"/api/projects/{project.slug}/training/run-toml-preview",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "[[directory]]" in body["dataset_toml"]
    assert 'type = "anima"' in body["run_toml"]
    assert body["launcher_argv"][0] == "deepspeed"


async def test_runs_list_empty(client, project: Project):
    resp = await client.get(f"/api/projects/{project.slug}/training/runs")
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


async def test_delete_unknown_run_404s(client, project: Project):
    resp = await client.delete(
        f"/api/projects/{project.slug}/training/runs/no-such-run",
    )
    assert resp.status_code == 404
