"""Smoke test that the `ui` command creates the app correctly."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from neme_anima.cli import app

runner = CliRunner()


def test_ui_help_lists_command():
    result = runner.invoke(app, ["--help"])
    assert "ui" in result.output


def test_ui_dry_run_starts_server_and_returns(tmp_path: Path, monkeypatch):
    """--dry-run constructs the app and returns 0 without entering uvicorn."""
    monkeypatch.setenv("NEME_STATE_DIR", str(tmp_path / "state"))
    result = runner.invoke(app, ["ui", "--dry-run"])
    assert result.exit_code == 0, result.output
