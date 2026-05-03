"""End-to-end CLI tests using typer.testing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from neme_anima.cli import app
from neme_anima.storage.project import Project

runner = CliRunner()


def test_project_create_makes_folder(tmp_path: Path):
    target = tmp_path / "newproj"
    result = runner.invoke(app, ["project", "create", str(target), "--name", "newproj"])
    assert result.exit_code == 0, result.output
    assert (target / "project.json").exists()
    p = Project.load(target)
    assert p.name == "newproj"


def test_project_add_video_appends_source(tmp_path: Path):
    Project.create(tmp_path / "p", name="p")
    fake = tmp_path / "ep01.mkv"
    fake.write_bytes(b"")
    result = runner.invoke(app, ["project", "add-video", str(tmp_path / "p"), str(fake)])
    assert result.exit_code == 0, result.output
    p = Project.load(tmp_path / "p")
    assert len(p.sources) == 1


def test_project_add_ref_appends(tmp_path: Path):
    Project.create(tmp_path / "p", name="p")
    img = tmp_path / "r.png"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(img)
    result = runner.invoke(app, ["project", "add-ref", str(tmp_path / "p"), str(img)])
    assert result.exit_code == 0, result.output
    p = Project.load(tmp_path / "p")
    assert len(p.refs) == 1


def test_project_create_rejects_existing(tmp_path: Path):
    target = tmp_path / "exists"
    target.mkdir()
    result = runner.invoke(app, ["project", "create", str(target), "--name", "x"])
    assert result.exit_code != 0


def _seed_src_with_character(src_root: Path) -> None:
    from neme_anima.storage.metadata import FrameRecord, MetadataLog
    from neme_anima.storage.project import Project

    p = Project.create(src_root, name="src")
    p.add_character(name="Sora", slug="sora")
    (p.kept_dir / "ep01__a.png").write_bytes(b"\x89PNG")
    MetadataLog(p.metadata_path).append(FrameRecord(
        filename="ep01__a", kept=True, scene_idx=0, tracklet_id=0, frame_idx=0,
        timestamp_seconds=0.0, bbox=(0, 0, 1, 1), ccip_distance=0.0,
        sharpness=0.0, visibility=0.0, aspect=1.0, score=0.0,
        video_stem="ep01", character_slug="sora",
    ))


def test_character_copy_cli(tmp_path):
    from neme_anima.storage.project import Project

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _seed_src_with_character(src)
    Project.create(dst, name="dst")

    res = subprocess.run(
        [sys.executable, "-m", "neme_anima.cli",
         "character", "copy", str(src), "sora", str(dst)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, (res.stdout + res.stderr)
    payload = json.loads(res.stdout)
    assert payload["character_slug"] == "sora"
    assert "ep01__a" in payload["frames_added"]

    dst_proj = Project.load(dst)
    assert dst_proj.character_by_slug("sora") is not None


def test_character_copy_cli_dry_run(tmp_path):
    from neme_anima.storage.project import Project

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    _seed_src_with_character(src)
    Project.create(dst, name="dst")

    res = subprocess.run(
        [sys.executable, "-m", "neme_anima.cli",
         "character", "copy", str(src), "sora", str(dst), "--dry-run"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, (res.stdout + res.stderr)
    payload = json.loads(res.stdout)
    assert payload["dry_run"] is True
    assert Project.load(dst).character_by_slug("sora") is None
