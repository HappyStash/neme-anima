"""End-to-end CLI tests using typer.testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from neme_extractor.cli import app
from neme_extractor.storage.project import Project

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
