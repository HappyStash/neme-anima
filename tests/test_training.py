"""Pure-function tests for the training module — TOML rendering, path
validation, and checkpoint pruning. The subprocess machinery in
``server.training_runner`` is not exercised here (it requires a real
diffusion-pipe install + GPU).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neme_anima import training
from neme_anima.storage.project import Project, TrainingConfig


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project.create(tmp_path / "p1", name="p1")


# ----- path validation ------------------------------------------------------


def test_check_path_empty():
    c = training.check_path("")
    assert not c.exists
    assert c.error and "empty" in c.error.lower()


def test_check_path_missing(tmp_path: Path):
    c = training.check_path(str(tmp_path / "nope.bin"), expect="file")
    assert not c.exists
    assert c.error and "no such" in c.error.lower()


def test_check_path_file_vs_dir(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    assert training.check_path(str(f), expect="file").error is None
    # Asking for a dir on a file must report an error.
    c = training.check_path(str(f), expect="dir")
    assert c.exists and c.error and "directory" in c.error.lower()


def test_validate_for_run_complains_about_missing_paths(project: Project):
    problems = training.validate_for_run(project.training)
    assert any("diffusion-pipe" in p for p in problems)
    assert any("DiT" in p or "transformer" in p for p in problems)
    assert any("VAE" in p for p in problems)
    assert any("text encoder" in p for p in problems)


def test_validate_for_run_passes_when_paths_resolve(
    project: Project, tmp_path: Path,
):
    # Build a fake diffusion-pipe install with a train.py + dummy weight files.
    dp_dir = tmp_path / "diffusion-pipe"
    dp_dir.mkdir()
    (dp_dir / "train.py").write_text("# fake")
    dit = tmp_path / "dit.safetensors"; dit.write_bytes(b"")
    vae = tmp_path / "vae.safetensors"; vae.write_bytes(b"")
    llm = tmp_path / "llm.safetensors"; llm.write_bytes(b"")

    cfg = project.training
    cfg.diffusion_pipe_dir = str(dp_dir)
    cfg.dit_path = str(dit)
    cfg.vae_path = str(vae)
    cfg.llm_path = str(llm)
    # Default launcher uses `deepspeed`, which isn't necessarily on the test
    # host's PATH. Override with a binary we know exists.
    cfg.launcher_override = "/bin/sh -c true {config}"
    assert training.validate_for_run(cfg) == []


def test_validate_for_run_requires_train_py(project: Project, tmp_path: Path):
    dp_dir = tmp_path / "diffusion-pipe-no-train"
    dp_dir.mkdir()
    # Note: no train.py
    dit = tmp_path / "dit.safetensors"; dit.write_bytes(b"")
    vae = tmp_path / "vae.safetensors"; vae.write_bytes(b"")
    llm = tmp_path / "llm.safetensors"; llm.write_bytes(b"")
    cfg = project.training
    cfg.diffusion_pipe_dir = str(dp_dir)
    cfg.dit_path = str(dit); cfg.vae_path = str(vae); cfg.llm_path = str(llm)
    problems = training.validate_for_run(cfg)
    assert any("train.py" in p for p in problems)


# ----- TOML rendering -------------------------------------------------------


def test_render_dataset_toml_defaults_to_kept_dir(project: Project):
    text = training.render_dataset_toml(project)
    assert "[[directory]]" in text
    assert str(project.kept_dir.resolve()) in text
    # AR bucket fields tracked from config.
    assert "enable_ar_bucket = true" in text
    assert "min_ar = 0.5" in text
    assert "num_ar_buckets = 9" in text
    # Default mixed resolutions.
    assert "[512, 1024]" in text


def test_render_dataset_toml_uses_passed_dataset_root(
    project: Project, tmp_path: Path,
):
    """The runner passes a per-run staging dir so the TOML must point at
    that path, not at kept_dir — otherwise diffusion-pipe would see the
    raw `_crop` derivatives as separate samples."""
    staged = tmp_path / "staged"
    staged.mkdir()
    text = training.render_dataset_toml(project, dataset_root=staged)
    assert str(staged.resolve()) in text
    assert str(project.kept_dir.resolve()) not in text


def test_render_run_toml_matches_reference_recipe(
    project: Project, tmp_path: Path,
):
    project.training.dit_path = str(tmp_path / "dit.bin")
    project.training.vae_path = str(tmp_path / "vae.bin")
    project.training.llm_path = str(tmp_path / "llm.bin")
    run_dir = tmp_path / "run"
    ds = tmp_path / "ds.toml"
    text = training.render_run_toml(
        project, run_dir=run_dir, dataset_toml_path=ds,
    )
    # Anima-specific knobs from the reference recipe.
    assert 'type = "anima"' in text
    assert "llm_adapter_lr = 0.0" in text
    assert "sigmoid_scale = 1.3" in text
    # LoRA adapter rank.
    assert "rank = 32" in text
    # Optimizer.
    assert 'type = "adamw_optimi"' in text
    assert "betas = [0.9, 0.99]" in text
    # No resume flag when not requested.
    assert "resume_from_checkpoint" not in text


def test_render_run_toml_includes_resume_flag(project: Project, tmp_path: Path):
    text = training.render_run_toml(
        project,
        run_dir=tmp_path / "run",
        dataset_toml_path=tmp_path / "ds.toml",
        resume_from_checkpoint="epoch20",
    )
    assert 'resume_from_checkpoint = "epoch20"' in text


def test_render_run_toml_emits_resume_at_top_level(
    project: Project, tmp_path: Path,
):
    """``resume_from_checkpoint`` must be a top-level key, not nested under a
    section. If it lands under [optimizer] (the previous bug), DeepSpeed
    forwards it to AdamW as a kwarg and training crashes with TypeError."""
    import tomllib
    project.training.dit_path = str(tmp_path / "dit.bin")
    project.training.vae_path = str(tmp_path / "vae.bin")
    project.training.llm_path = str(tmp_path / "llm.bin")
    text = training.render_run_toml(
        project,
        run_dir=tmp_path / "run",
        dataset_toml_path=tmp_path / "ds.toml",
        resume_from_checkpoint="20260501_11-51-58",
    )
    parsed = tomllib.loads(text)
    assert parsed.get("resume_from_checkpoint") == "20260501_11-51-58"
    # Sanity: not bleeding into any subsection.
    for section in ("optimizer", "adapter", "model"):
        if section in parsed:
            assert "resume_from_checkpoint" not in parsed[section]


def test_run_toml_quotes_paths_with_special_chars(
    project: Project, tmp_path: Path,
):
    weird = tmp_path / 'name with "quotes".bin'
    weird.write_bytes(b"")
    project.training.dit_path = str(weird)
    text = training.render_run_toml(
        project, run_dir=tmp_path, dataset_toml_path=tmp_path / "ds.toml",
    )
    # Embedded double-quotes must be escaped, not bare.
    assert r'\"quotes\"' in text


# ----- dataset staging ------------------------------------------------------


def _png(path: Path, value: int = 0) -> None:
    """Write a 4×4 solid-color PNG at ``path``."""
    from PIL import Image
    import numpy as np
    Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(path)


def test_build_dataset_staging_pairs_originals(project: Project, tmp_path: Path):
    """Frames without a crop are staged as plain symlinks to the original
    image and its sidecar."""
    _png(project.kept_dir / "f1.png")
    (project.kept_dir / "f1.txt").write_text("tag_a, tag_b\n", encoding="utf-8")

    dest = tmp_path / "ds"
    info = training.build_dataset_staging(project, dest)
    assert info["images"] == 1
    assert info["with_crop"] == 0
    assert info["missing_txt"] == 0

    # Both pair members exist at the staging path.
    assert (dest / "f1.png").exists()
    assert (dest / "f1.txt").exists()
    # Sidecar content reads back through the link.
    assert (dest / "f1.txt").read_text(encoding="utf-8") == "tag_a, tag_b\n"


def test_build_dataset_staging_substitutes_crop_image(
    project: Project, tmp_path: Path,
):
    """When a `_crop` derivative exists, the staged image points at the
    crop's pixels but the sidecar still points at the original `.txt`.
    This is the on-disk realization of "edit tags on the original; train
    on the crop"."""
    _png(project.kept_dir / "f1.png", value=200)        # original (light)
    _png(project.kept_dir / "f1_crop.png", value=50)    # crop (dark)
    (project.kept_dir / "f1.txt").write_text("orig_tags\n", encoding="utf-8")

    dest = tmp_path / "ds"
    info = training.build_dataset_staging(project, dest)
    assert info["images"] == 1
    assert info["with_crop"] == 1
    # The trainer sees `f1.png`, not `f1_crop.png` — pairing is by stem.
    assert sorted(p.name for p in dest.iterdir()) == ["f1.png", "f1.txt"]
    # The staged image's bytes are the crop's (dark pixels).
    from PIL import Image
    import numpy as np
    with Image.open(dest / "f1.png") as im:
        assert int(np.array(im).mean()) < 100
    # The staged sidecar's content is the original's tags.
    assert (dest / "f1.txt").read_text(encoding="utf-8") == "orig_tags\n"


def test_build_dataset_staging_ignores_legacy_crop_txt(
    project: Project, tmp_path: Path,
):
    """A leftover `<name>_crop.txt` from older project layouts must not
    leak into the trainer's view — only the original sidecar is staged."""
    _png(project.kept_dir / "f1.png")
    _png(project.kept_dir / "f1_crop.png")
    (project.kept_dir / "f1.txt").write_text("orig\n", encoding="utf-8")
    (project.kept_dir / "f1_crop.txt").write_text("legacy\n", encoding="utf-8")

    dest = tmp_path / "ds"
    training.build_dataset_staging(project, dest)
    assert (dest / "f1.txt").read_text(encoding="utf-8") == "orig\n"
    # No `f1_crop.*` shadow files in the staging dir.
    assert not any(p.name.endswith("_crop.png") for p in dest.iterdir())
    assert not any(p.name.endswith("_crop.txt") for p in dest.iterdir())


def test_build_dataset_staging_rebuilds_dest(
    project: Project, tmp_path: Path,
):
    """Re-running staging must wipe stale links so a removed/renamed
    frame doesn't linger in the trainer's view."""
    _png(project.kept_dir / "f1.png")
    (project.kept_dir / "f1.txt").write_text("a\n", encoding="utf-8")

    dest = tmp_path / "ds"
    training.build_dataset_staging(project, dest)
    assert (dest / "f1.png").exists()

    # Drop f1; add f2. The next staging pass should reflect that.
    (project.kept_dir / "f1.png").unlink()
    (project.kept_dir / "f1.txt").unlink()
    _png(project.kept_dir / "f2.png")
    (project.kept_dir / "f2.txt").write_text("b\n", encoding="utf-8")

    info = training.build_dataset_staging(project, dest)
    assert info["images"] == 1
    assert not (dest / "f1.png").exists()
    assert (dest / "f2.png").exists()


def test_build_dataset_staging_counts_missing_txt(
    project: Project, tmp_path: Path,
):
    """A sidecar-less image is still staged (the trainer can train on it
    if the user is OK with empty captions) but the count surfaces so the
    caller can warn."""
    _png(project.kept_dir / "f1.png")
    # No f1.txt on disk.
    info = training.build_dataset_staging(project, tmp_path / "ds")
    assert info["images"] == 1
    assert info["missing_txt"] == 1


# ----- launcher argv --------------------------------------------------------


def test_default_launcher_argv(project: Project, tmp_path: Path):
    run_toml = tmp_path / "run.toml"
    argv = training.build_launcher_argv(project.training, run_toml=run_toml)
    # First token is the launcher binary, possibly resolved to an absolute
    # path if 'deepspeed' is in PATH or the diffusion-pipe venv.
    assert argv[0].endswith("deepspeed")
    assert "--num_gpus=1" in argv
    assert str(run_toml.resolve()) in argv


def test_launcher_override_with_placeholder(
    project: Project, tmp_path: Path,
):
    # Use a binary we know exists at a stable path so the resolution is
    # deterministic across hosts.
    project.training.launcher_override = "/bin/sh wrapper.py {config}"
    run_toml = tmp_path / "run.toml"
    argv = training.build_launcher_argv(project.training, run_toml=run_toml)
    assert argv == ["/bin/sh", "wrapper.py", str(run_toml.resolve())]


def test_launcher_override_without_placeholder_appends(
    project: Project, tmp_path: Path,
):
    project.training.launcher_override = "/bin/sh wrapper.py"
    run_toml = tmp_path / "run.toml"
    argv = training.build_launcher_argv(project.training, run_toml=run_toml)
    assert argv == ["/bin/sh", "wrapper.py", str(run_toml.resolve())]


def test_launcher_resolves_via_diffusion_pipe_venv(
    project: Project, tmp_path: Path,
):
    """A binary in <diffusion_pipe_dir>/.venv/bin should be discovered even
    when not on the system PATH."""
    dp = tmp_path / "dp"
    venv_bin = dp / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    fake = venv_bin / "fake-launcher"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    project.training.diffusion_pipe_dir = str(dp)
    project.training.launcher_override = "fake-launcher {config}"
    run_toml = tmp_path / "run.toml"
    argv = training.build_launcher_argv(project.training, run_toml=run_toml)
    assert argv[0] == str(fake)


# ----- checkpoint discovery + retention -------------------------------------


def _make_ckpt(parent: Path, name: str) -> Path:
    p = parent / name
    p.mkdir(parents=True)
    (p / "weights.safetensors").write_bytes(b"x" * 1024)
    return p


def test_discover_checkpoints_sorts_by_epoch(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_ckpt(run_dir, "epoch10")
    _make_ckpt(run_dir, "epoch2")
    _make_ckpt(run_dir, "epoch5")
    cps = training.discover_checkpoints(run_dir)
    assert [c.name for c in cps] == ["epoch2", "epoch5", "epoch10"]
    for c in cps:
        assert c.size_bytes >= 1024


def test_discover_checkpoints_handles_global_step(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_ckpt(run_dir, "global_step1000")
    _make_ckpt(run_dir, "global_step500")
    cps = training.discover_checkpoints(run_dir)
    assert [c.name for c in cps] == ["global_step500", "global_step1000"]


def test_discover_checkpoints_ignores_non_ckpt_dirs(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_ckpt(run_dir, "epoch1")
    (run_dir / "logs").mkdir()
    (run_dir / "config.json").write_text("{}")
    cps = training.discover_checkpoints(run_dir)
    assert [c.name for c in cps] == ["epoch1"]


def test_discover_checkpoints_finds_nested_diffusion_pipe_layout(tmp_path: Path):
    """diffusion-pipe writes its checkpoints into a per-launch timestamped
    subdirectory under ``output_dir``, not directly into ``output_dir``."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sub = run_dir / "20260501_11-51-58"
    sub.mkdir()
    _make_ckpt(sub, "epoch10")
    _make_ckpt(sub, "epoch20")
    _make_ckpt(sub, "global_step720")
    cps = training.discover_checkpoints(run_dir)
    assert {c.name for c in cps} == {"epoch10", "epoch20", "global_step720"}
    for c in cps:
        assert c.subdir == "20260501_11-51-58"
        assert str(sub) in c.path


def test_discover_checkpoints_skips_dataset_dir(tmp_path: Path):
    """The staged training dataset lives under run_dir/dataset/ and must
    never be treated as a checkpoint container."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "dataset").mkdir()
    # If we were to recurse here we'd find nothing, but we still want to
    # avoid the wasted scan: just assert no spurious checkpoints surface.
    sub = run_dir / "20260501_11-51-58"
    sub.mkdir()
    _make_ckpt(sub, "epoch1")
    cps = training.discover_checkpoints(run_dir)
    assert [c.name for c in cps] == ["epoch1"]


def test_find_resumable_subdir_returns_subdir_with_latest_marker(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sub = run_dir / "20260501_11-51-58"
    sub.mkdir()
    (sub / "latest").write_text("global_step720")
    assert training.find_resumable_subdir(run_dir) == "20260501_11-51-58"


def test_find_resumable_subdir_picks_most_recent(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    older = run_dir / "20260501_11-51-58"
    older.mkdir()
    (older / "latest").write_text("global_step100")
    newer = run_dir / "20260501_14-22-10"
    newer.mkdir()
    newer_latest = newer / "latest"
    newer_latest.write_text("global_step200")
    # Force mtime ordering so the assertion is robust on fast filesystems.
    import os, time
    os.utime(older / "latest", (time.time() - 60, time.time() - 60))
    assert training.find_resumable_subdir(run_dir) == "20260501_14-22-10"


def test_find_resumable_subdir_none_when_no_latest(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    sub = run_dir / "20260501_11-51-58"
    sub.mkdir()
    # No 'latest' marker = nothing to resume from.
    assert training.find_resumable_subdir(run_dir) is None


def test_prune_keeps_all_when_zero(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for i in (1, 2, 3, 4):
        _make_ckpt(run_dir, f"epoch{i}")
    deleted = training.prune_checkpoints(run_dir, keep_last_n=0)
    assert deleted == []
    assert len(training.discover_checkpoints(run_dir)) == 4


def test_prune_keeps_last_n(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for i in (1, 2, 3, 4, 5):
        _make_ckpt(run_dir, f"epoch{i}")
    deleted = training.prune_checkpoints(run_dir, keep_last_n=2)
    assert sorted(deleted) == ["epoch1", "epoch2", "epoch3"]
    remaining = [c.name for c in training.discover_checkpoints(run_dir)]
    assert remaining == ["epoch4", "epoch5"]


def test_prune_no_op_when_under_limit(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_ckpt(run_dir, "epoch1")
    _make_ckpt(run_dir, "epoch2")
    assert training.prune_checkpoints(run_dir, keep_last_n=5) == []


# ----- caption rendering ----------------------------------------------------


def test_render_training_caption_modes():
    cfg = TrainingConfig()
    cfg.caption_mode = "tags"
    assert training.render_training_caption(tags="1girl, blue eyes", nl="A girl smiles.", config=cfg) == "1girl, blue eyes"

    cfg.caption_mode = "nl"
    assert training.render_training_caption(tags="1girl", nl="A girl.", config=cfg) == "A girl."

    cfg.caption_mode = "mixed"
    assert training.render_training_caption(tags="1girl", nl="A girl.", config=cfg) == "1girl. A girl."


def test_render_training_caption_with_trigger():
    cfg = TrainingConfig()
    cfg.trigger_token = "mychar"
    cfg.caption_mode = "mixed"
    out = training.render_training_caption(tags="1girl", nl="A girl.", config=cfg)
    assert out == "mychar, 1girl. A girl."


# ----- run-dir helpers + project storage round-trip ------------------------


def test_new_run_dir_creates_unique_dirs(project: Project):
    a = training.new_run_dir(project, label="x")
    b = training.new_run_dir(project, label="x")
    assert a.exists() and b.exists()
    assert a != b


def test_training_config_round_trips(project: Project):
    project.training.dit_path = "/foo/dit.safetensors"
    project.training.keep_last_n_checkpoints = 5
    project.training.preset = "character"
    project.training.learning_rate = 5e-5
    project.training.resolutions = [768, 1024]
    project.save()

    reloaded = Project.load(project.root)
    assert reloaded.training.dit_path == "/foo/dit.safetensors"
    assert reloaded.training.keep_last_n_checkpoints == 5
    assert reloaded.training.preset == "character"
    assert reloaded.training.learning_rate == 5e-5
    assert reloaded.training.resolutions == [768, 1024]


def test_training_config_default_round_trip_keep_all(project: Project):
    project.save()
    reloaded = Project.load(project.root)
    # The user-requested default is 0 — keep all checkpoints.
    assert reloaded.training.keep_last_n_checkpoints == 0
