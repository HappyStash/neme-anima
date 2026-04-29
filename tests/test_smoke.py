"""Smoke test: package imports and CLI is registered."""

from __future__ import annotations


def test_package_imports():
    import neme_extractor

    assert neme_extractor.__version__


def test_cli_imports():
    from neme_extractor.cli import app

    assert app is not None


def test_config_roundtrip(tmp_path):
    from neme_extractor.config import Thresholds

    t = Thresholds()
    p = tmp_path / "thresholds.json"
    t.to_json(p)
    t2 = Thresholds.from_json(p)
    assert t2.detect.person_score_min == t.detect.person_score_min
