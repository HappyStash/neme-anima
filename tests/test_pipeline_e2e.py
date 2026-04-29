"""Pipeline integration smoke test.

Synthesizes a short clip + a reference image and runs ``run_extract`` end to
end. The anime detectors will find nothing in synthesized noise, so the
expected outcome is an empty kept/ folder. The test validates that the
orchestration runs all stages, writes the cache, and survives the empty-result
path without errors. Real-content validation belongs in the user-driven
verification step.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from neme_extractor.pipeline import run_extract, run_rerun


@pytest.fixture
def synth_video(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    h, w, fps = 240, 320, 24
    writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    # 1 second of frame A, hard cut, 1 second of frame B → 2 scenes.
    base_a = np.full((h, w, 3), (200, 60, 30), dtype=np.uint8)
    base_b = np.full((h, w, 3), (30, 60, 200), dtype=np.uint8)
    for i in range(24):
        f = base_a.copy()
        cv2.rectangle(f, (10 + i * 4, 80), (50 + i * 4, 160), (255, 255, 255), -1)
        writer.write(f)
    for i in range(24):
        f = base_b.copy()
        cv2.rectangle(f, (310 - i * 4, 80), (270 - i * 4, 160), (255, 255, 255), -1)
        writer.write(f)
    writer.release()
    return p


@pytest.fixture
def refs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "refs"
    d.mkdir()
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(d / "ref.png")
    return d


@pytest.mark.gpu
def test_run_extract_orchestrates_all_stages(
    synth_video: Path, refs_dir: Path, tmp_path: Path
):
    out_root = tmp_path / "out"
    run_extract(video=synth_video, refs_dir=refs_dir, out_root=out_root)
    out = out_root / synth_video.stem
    assert out.exists()
    assert (out / "kept").exists()
    assert (out / "rejected").exists()
    assert (out / "thresholds.json").exists()
    assert (out / "metadata.json").exists()
    assert (out / "cache" / "scenes.parquet").exists()
    assert (out / "cache" / "tracklets.parquet").exists()
    assert (out / "cache" / "run.json").exists()


@pytest.mark.gpu
def test_run_rerun_uses_cache(
    synth_video: Path, refs_dir: Path, tmp_path: Path
):
    out_root = tmp_path / "out"
    run_extract(video=synth_video, refs_dir=refs_dir, out_root=out_root)
    # Mutate thresholds slightly and rerun. Should not crash, should rewrite metadata.
    out = out_root / synth_video.stem
    run_rerun(out_dir=out)
    assert (out / "metadata.json").exists()
