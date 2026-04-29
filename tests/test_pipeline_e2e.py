"""Pipeline integration smoke test (project-centric)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from neme_extractor.pipeline import run_extract, run_rerun
from neme_extractor.storage.project import Project


@pytest.fixture
def synth_video(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    h, w, fps = 240, 320, 24
    writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
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
def ref_image(tmp_path: Path) -> Path:
    p = tmp_path / "ref.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(p)
    return p


@pytest.mark.gpu
def test_run_extract_project_centric(synth_video: Path, ref_image: Path, tmp_path: Path):
    project = Project.create(tmp_path / "proj", name="proj")
    project.add_ref(ref_image)
    project.add_source(synth_video)

    run_extract(project=project, source_idx=0)

    # Project-rooted output exists.
    assert (project.kept_dir).exists()
    assert (project.rejected_dir).exists()
    # Per-video cache is under cache/<video_stem>/.
    assert (project.cache_dir_for("clip")).exists()
    assert (project.cache_dir_for("clip") / "scenes.parquet").exists()
    assert (project.cache_dir_for("clip") / "tracklets.parquet").exists()


@pytest.mark.gpu
def test_run_rerun_uses_cache(synth_video: Path, ref_image: Path, tmp_path: Path):
    project = Project.create(tmp_path / "proj", name="proj")
    project.add_ref(ref_image)
    project.add_source(synth_video)
    run_extract(project=project, source_idx=0)
    run_rerun(project=project, video_stem="clip")
    # Still works.
    assert (project.cache_dir_for("clip") / "tracklets.parquet").exists()


@pytest.mark.gpu
def test_run_extract_respects_excluded_refs(
    synth_video: Path, ref_image: Path, tmp_path: Path
):
    """If all refs are excluded for a source, the extraction must fail clearly."""
    project = Project.create(tmp_path / "proj", name="proj")
    project.add_ref(ref_image)
    project.add_source(synth_video)
    project.set_excluded_refs(0, [str(Path(ref_image).resolve())])
    with pytest.raises(ValueError, match="no effective references"):
        run_extract(project=project, source_idx=0)
