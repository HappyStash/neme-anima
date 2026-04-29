"""Smoke tests for tag.py — WD14 runs end-to-end and writes kohya .txt sidecars."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from neme_extractor.config import TagConfig
from neme_extractor.tag import Tagger, write_tags_sidecar


def test_write_tags_sidecar_produces_txt(tmp_path: Path):
    img_path = tmp_path / "frame_001.png"
    img_path.write_bytes(b"fake")
    txt = write_tags_sidecar(img_path, "1girl, smile, blue_hair")
    assert txt == tmp_path / "frame_001.txt"
    assert txt.read_text(encoding="utf-8") == "1girl, smile, blue_hair\n"


def test_compose_text_excludes_filtered_tags():
    cfg = TagConfig(exclude_tags=("simple_background",))
    tagger = Tagger(cfg)
    text = tagger._compose_text(
        general={"smile": 0.9, "simple_background": 0.95, "blue_hair": 0.8},
        character={},
    )
    parts = [p.strip() for p in text.split(",")]
    assert "simple_background" not in parts
    assert "smile" in parts and "blue_hair" in parts


def test_compose_text_orders_character_first():
    tagger = Tagger(TagConfig())
    text = tagger._compose_text(
        general={"smile": 0.95, "blue_hair": 0.9},
        character={"some_character": 0.99},
    )
    assert text.startswith("some_character")


@pytest.mark.gpu
def test_tag_runs_end_to_end(tmp_path: Path):
    """Triggers a one-time WD14 weights download and runs on a noise image.
    The tag set may be empty / nonsensical (it's noise) but the call must not crash.
    """
    rng = np.random.default_rng(0)
    img_arr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)

    tagger = Tagger(TagConfig(model_name="EVA02_Large"))
    res = tagger.tag(img_arr)
    assert isinstance(res.text, str)
    assert isinstance(res.general, dict)
    assert isinstance(res.character, dict)

    img_path = tmp_path / "test.png"
    Image.fromarray(img_arr).save(img_path)
    txt = write_tags_sidecar(img_path, res.text)
    assert txt.exists()
    assert txt.suffix == ".txt"
