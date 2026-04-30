"""Pure-function tests for the training module — TOML rendering, path
validation, and checkpoint pruning. The subprocess machinery in
``server.training_runner`` is not exercised here (it requires a real
diffusion-pipe install + GPU).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neme_extractor import training
from neme_extractor.storage.project import Project, TrainingConfig


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


def test_render_dataset_toml_includes_kept_dir(project: Project):
    text = training.render_dataset_toml(project)
    assert "[[directory]]" in text
    assert str(project.kept_dir.resolve()) in text
    # AR bucket fields tracked from config.
    assert "enable_ar_bucket = true" in text
    assert "min_ar = 0.5" in text
    assert "num_ar_buckets = 9" in text
    # Default mixed resolutions.
    assert "[512, 1024]" in text


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


# ----- launcher argv --------------------------------------------------------


def test_default_launcher_argv(project: Project, tmp_path: Path):
    run_toml = tmp_path / "run.toml"
    argv = training.build_launcher_argv(project.training, run_toml=run_toml)
    assert argv[0] == "deepspeed"
    assert "--num_gpus=1" in argv
    assert str(run_toml.resolve()) in argv


def test_launcher_override_with_placeholder(
    project: Project, tmp_path: Path,
):
    project.training.launcher_override = "python wrapper.py {config}"
    run_toml = tmp_path / "run.toml"
    argv = training.build_launcher_argv(project.training, run_toml=run_toml)
    assert argv == ["python", "wrapper.py", str(run_toml.resolve())]


def test_launcher_override_without_placeholder_appends(
    project: Project, tmp_path: Path,
):
    project.training.launcher_override = "python wrapper.py"
    run_toml = tmp_path / "run.toml"
    argv = training.build_launcher_argv(project.training, run_toml=run_toml)
    assert argv == ["python", "wrapper.py", str(run_toml.resolve())]


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
