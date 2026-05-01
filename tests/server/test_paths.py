"""Tests for `neme_anima.server.paths.normalize_input_path`."""

from __future__ import annotations

from pathlib import Path

import pytest

from neme_anima.server import paths as pmod


def test_plain_posix_path_unchanged(tmp_path: Path):
    assert pmod.normalize_input_path(str(tmp_path / "x.mkv")) == tmp_path / "x.mkv"


def test_strips_surrounding_quotes(tmp_path: Path):
    assert pmod.normalize_input_path(f'"{tmp_path}/x.mkv"') == tmp_path / "x.mkv"


def test_rejects_vfs_sentinel():
    with pytest.raises(ValueError, match="paste the absolute path"):
        pmod.normalize_input_path("vfs://Classroom of the Elite - S03E11 [1080p].mkv")


def test_windows_drive_backslash_on_wsl(monkeypatch):
    monkeypatch.setattr(pmod, "is_wsl", lambda: True)
    out = pmod.normalize_input_path(r"C:\Users\me\foo.mkv")
    assert out == Path("/mnt/c/Users/me/foo.mkv")


def test_windows_drive_forward_slash_on_wsl(monkeypatch):
    monkeypatch.setattr(pmod, "is_wsl", lambda: True)
    out = pmod.normalize_input_path("D:/Videos/Show E01.mkv")
    assert out == Path("/mnt/d/Videos/Show E01.mkv")


def test_file_uri_with_windows_drive_on_wsl(monkeypatch):
    monkeypatch.setattr(pmod, "is_wsl", lambda: True)
    out = pmod.normalize_input_path(
        "file:///C:/Users/me/Show%20-%20S03E11%20%5B1080p%5D.mkv"
    )
    assert out == Path("/mnt/c/Users/me/Show - S03E11 [1080p].mkv")


def test_file_uri_posix(monkeypatch):
    monkeypatch.setattr(pmod, "is_wsl", lambda: True)
    out = pmod.normalize_input_path("file:///home/me/Show%20E01.mkv")
    assert out == Path("/home/me/Show E01.mkv")


def test_windows_drive_off_wsl_kept_as_drive(monkeypatch):
    monkeypatch.setattr(pmod, "is_wsl", lambda: False)
    out = pmod.normalize_input_path(r"C:\Users\me\foo.mkv")
    # Off WSL we don't translate; just normalize separators / preserve the drive.
    assert str(out).lower().startswith("c:")
