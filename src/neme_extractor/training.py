"""Anima LoRA-training helpers.

This module is **pure plumbing**: it knows how to validate the user's
configuration, render dataset/run TOML files for tdrussell/diffusion-pipe,
discover checkpoints on disk, and prune old ones according to retention
policy. The actual training subprocess lives in
``neme_extractor.server.training_runner``.

The TOML output mirrors the reference recipe documented in
``docs/anima-lora-training-notes.md``. We deliberately avoid a TOML library
dependency — the schema is simple, stable, and easier to read in plain
strings than rebuilt through ``tomli_w``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from neme_extractor.storage.project import Project, TrainingConfig

# ----- caption defaults -----------------------------------------------------

# 2-sentence quality prefix from the Anima model card (tag-ordered structure).
DEFAULT_QUALITY_PREFIX = "masterpiece, best quality, score_7, safe,"

# Caption suffix for the natural-language portion when the user has no LLM
# description on disk yet. Empty string => no NL appended.
DEFAULT_NL_FALLBACK = ""


# ----- dataclasses ----------------------------------------------------------


@dataclass
class CheckpointInfo:
    """One on-disk training checkpoint produced by diffusion-pipe."""

    name: str               # e.g. "epoch20" or "global_step5000"
    path: str               # absolute path to the directory
    epoch: int | None       # parsed epoch number if the name encodes it
    step: int | None        # parsed global step if encoded
    size_bytes: int         # cumulative size on disk (best-effort)
    modified_at: str        # ISO-8601


@dataclass
class PathCheck:
    path: str
    exists: bool
    is_file: bool
    is_dir: bool
    error: str | None = None


# ----- path validation -------------------------------------------------------


def check_path(raw: str, *, expect: str = "any") -> PathCheck:
    """Stat ``raw`` and report what it is.

    ``expect`` ∈ {"any", "file", "dir"} — if set, ``error`` is populated when
    the resolved path doesn't match. Empty/whitespace input returns
    ``exists=False`` with a friendly error so the UI can render a red X
    without an exception.
    """
    raw = (raw or "").strip()
    if not raw:
        return PathCheck(
            path=raw, exists=False, is_file=False, is_dir=False,
            error="path is empty",
        )
    p = Path(raw).expanduser()
    try:
        exists = p.exists()
    except OSError as e:
        return PathCheck(
            path=str(p), exists=False, is_file=False, is_dir=False, error=str(e),
        )
    is_file = p.is_file()
    is_dir = p.is_dir()
    error: str | None = None
    if not exists:
        error = "no such file or directory"
    elif expect == "file" and not is_file:
        error = "expected a file, got a directory"
    elif expect == "dir" and not is_dir:
        error = "expected a directory, got a file"
    return PathCheck(
        path=str(p), exists=exists, is_file=is_file, is_dir=is_dir, error=error,
    )


def validate_for_run(config: TrainingConfig) -> list[str]:
    """Return a list of human-readable problems blocking a training run.

    Empty list = good to go. Used at the API layer to gate POST /start.
    """
    problems: list[str] = []
    checks = {
        "diffusion-pipe directory": check_path(
            config.diffusion_pipe_dir, expect="dir",
        ),
        "Anima DiT (transformer) file": check_path(
            config.dit_path, expect="file",
        ),
        "Qwen image VAE file": check_path(config.vae_path, expect="file"),
        "Qwen 3 0.6B base text encoder file": check_path(
            config.llm_path, expect="file",
        ),
    }
    for label, c in checks.items():
        if c.error:
            problems.append(f"{label}: {c.error} ({c.path or '<empty>'})")

    train_py = Path(config.diffusion_pipe_dir).expanduser() / "train.py"
    if config.diffusion_pipe_dir and not train_py.is_file():
        problems.append(
            f"diffusion-pipe directory does not contain train.py at {train_py}",
        )
    if not config.resolutions:
        problems.append("at least one training resolution is required")
    if config.epochs <= 0:
        problems.append("epochs must be > 0")
    if config.rank <= 0:
        problems.append("rank must be > 0")
    if config.learning_rate <= 0:
        problems.append("learning_rate must be > 0")
    return problems


# ----- TOML generation -------------------------------------------------------


def _toml_value(v: object) -> str:
    """Tiny TOML emitter — handles the subset we use (int/float/bool/str/list)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        # Avoid scientific-notation floats for clarity — diffusion-pipe is
        # tolerant of either, but readable TOML is nicer for the user.
        if isinstance(v, float):
            return repr(v)
        return str(v)
    if isinstance(v, str):
        # Use double-quoted strings, escape backslashes/double-quotes.
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"unsupported TOML value type: {type(v).__name__}")


def _toml_kv(key: str, value: object) -> str:
    return f"{key} = {_toml_value(value)}"


def render_dataset_toml(project: Project) -> str:
    """Build the dataset.toml that points diffusion-pipe at the project's
    ``output/kept`` folder.

    The kept/ directory holds image + ``.txt`` sidecar pairs (the trainer's
    expected format); we just point at it directly. AR bucketing values come
    from the project's ``TrainingConfig`` so ``num_ar_buckets`` and
    ``min_ar``/``max_ar`` stay in sync with the run config.
    """
    cfg = project.training
    lines = [
        "# Auto-generated by neme-extractor — do not hand-edit.",
        f'# Project: {project.slug}',
        "",
        "resolutions = " + _toml_value(cfg.resolutions),
        _toml_kv("enable_ar_bucket", cfg.enable_ar_bucket),
        _toml_kv("min_ar", cfg.min_ar),
        _toml_kv("max_ar", cfg.max_ar),
        _toml_kv("num_ar_buckets", cfg.num_ar_buckets),
        "",
        "[[directory]]",
        _toml_kv("path", str(project.kept_dir.resolve())),
        _toml_kv("num_repeats", 1),
        "",
    ]
    return "\n".join(lines)


def render_run_toml(
    project: Project,
    *,
    run_dir: Path,
    dataset_toml_path: Path,
    resume_from_checkpoint: str | None = None,
) -> str:
    """Build the run TOML for ``train.py --config``.

    ``resume_from_checkpoint`` is the *name* (not full path) of a checkpoint
    directory under ``run_dir`` — diffusion-pipe resolves resume paths
    relative to ``output_dir``.
    """
    cfg = project.training
    lines = [
        "# Auto-generated by neme-extractor — do not hand-edit.",
        f"# Project: {project.slug}",
        f"# Generated at {datetime.now(timezone.utc).isoformat()}",
        "",
        _toml_kv("output_dir", str(run_dir.resolve())),
        _toml_kv("dataset", str(dataset_toml_path.resolve())),
        "",
        # Training settings
        _toml_kv("epochs", cfg.epochs),
        _toml_kv("micro_batch_size_per_gpu", cfg.micro_batch_size),
        _toml_kv("pipeline_stages", 1),
        _toml_kv("gradient_accumulation_steps", cfg.gradient_accumulation_steps),
        _toml_kv("gradient_clipping", cfg.gradient_clipping),
        _toml_kv("warmup_steps", cfg.warmup_steps),
        "",
        # Eval
        _toml_kv("eval_every_n_epochs", cfg.eval_every_n_epochs),
        _toml_kv("eval_before_first_step", True),
        _toml_kv("eval_micro_batch_size_per_gpu", 1),
        _toml_kv("eval_gradient_accumulation_steps", 1),
        "",
        # Checkpointing
        _toml_kv("save_every_n_epochs", cfg.save_every_n_epochs),
        _toml_kv("activation_checkpointing", True),
        _toml_kv("partition_method", "parameters"),
        _toml_kv("save_dtype", "bfloat16"),
        _toml_kv("caching_batch_size", 1),
        _toml_kv("map_num_proc", 8),
        _toml_kv("steps_per_print", 1),
        _toml_kv("compile", True),
        "",
        "[model]",
        _toml_kv("type", "anima"),
        _toml_kv("transformer_path", str(Path(cfg.dit_path).expanduser().resolve())),
        _toml_kv("vae_path", str(Path(cfg.vae_path).expanduser().resolve())),
        _toml_kv("llm_path", str(Path(cfg.llm_path).expanduser().resolve())),
        _toml_kv("dtype", "bfloat16"),
        _toml_kv("llm_adapter_lr", cfg.llm_adapter_lr),
        _toml_kv("sigmoid_scale", cfg.sigmoid_scale),
        "",
        "[adapter]",
        _toml_kv("type", "lora"),
        _toml_kv("rank", cfg.rank),
        _toml_kv("dtype", "bfloat16"),
        "",
        "[optimizer]",
        _toml_kv("type", "adamw_optimi"),
        _toml_kv("lr", cfg.learning_rate),
        _toml_kv("betas", cfg.optimizer_betas),
        _toml_kv("weight_decay", cfg.weight_decay),
        _toml_kv("eps", cfg.eps),
    ]
    if resume_from_checkpoint:
        lines += [
            "",
            _toml_kv("resume_from_checkpoint", resume_from_checkpoint),
        ]
    lines.append("")
    return "\n".join(lines)


# ----- caption preview -------------------------------------------------------


def _read_sidecar(image_path: Path) -> tuple[str, str]:
    """Return (tag_line, nl_line) from the ``.txt`` sidecar of an image.

    The pipeline writes one or two lines: line 1 = comma-separated tags,
    line 2 (optional) = LLM natural-language description. Missing file =
    both empty.
    """
    sidecar = image_path.with_suffix(".txt")
    if not sidecar.is_file():
        return ("", "")
    try:
        text = sidecar.read_text(encoding="utf-8")
    except OSError:
        return ("", "")
    parts = text.splitlines()
    tags = parts[0].strip() if len(parts) >= 1 else ""
    nl = parts[1].strip() if len(parts) >= 2 else ""
    return (tags, nl)


def render_training_caption(
    *, tags: str, nl: str, config: TrainingConfig,
) -> str:
    """Render the caption that the trainer will see for one image.

    This does NOT mutate the on-disk sidecars — the trainer reads them as-is.
    The function is for *previewing* what the chosen ``caption_mode`` and
    ``trigger_token`` will produce so the user can sanity-check before
    starting a run.
    """
    trigger = (config.trigger_token or "").strip()
    tags = tags.strip()
    nl = nl.strip()
    if config.caption_mode == "tags":
        body = tags
    elif config.caption_mode == "nl":
        body = nl
    else:  # "mixed"
        if tags and nl:
            body = f"{tags}. {nl}"
        else:
            body = tags or nl
    if trigger:
        return f"{trigger}, {body}".rstrip(", ")
    return body


def dataset_preview(project: Project, *, sample_n: int = 5) -> dict:
    """Summarize the project's training dataset for the UI."""
    cfg = project.training
    kept = project.kept_dir
    if not kept.is_dir():
        return {
            "total_images": 0, "with_tags": 0, "with_descriptions": 0,
            "samples": [], "kept_dir": str(kept),
        }
    images = sorted(
        p for p in kept.iterdir()
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    total = len(images)
    with_tags = 0
    with_nl = 0
    samples: list[dict] = []
    for i, img in enumerate(images):
        tags, nl = _read_sidecar(img)
        if tags:
            with_tags += 1
        if nl:
            with_nl += 1
        if i < sample_n:
            samples.append({
                "filename": img.name,
                "tags": tags,
                "nl": nl,
                "rendered": render_training_caption(tags=tags, nl=nl, config=cfg),
            })
    return {
        "total_images": total,
        "with_tags": with_tags,
        "with_descriptions": with_nl,
        "samples": samples,
        "kept_dir": str(kept),
    }


# ----- checkpoint discovery + retention --------------------------------------

# diffusion-pipe writes per-epoch directories named ``epoch{N}`` under the
# run's output_dir; some forks also use ``global_step{N}``. Match either.
_CKPT_PATTERNS = (
    re.compile(r"^epoch(\d+)$"),
    re.compile(r"^global_step(\d+)$"),
    re.compile(r"^step(\d+)$"),
)


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _parse_checkpoint_name(name: str) -> tuple[int | None, int | None]:
    """Return (epoch, step) parsed from a checkpoint directory name.

    Either component may be None depending on the trainer's naming scheme.
    """
    for pat in _CKPT_PATTERNS:
        m = pat.match(name)
        if m:
            n = int(m.group(1))
            if pat.pattern.startswith("^epoch"):
                return (n, None)
            return (None, n)
    return (None, None)


def discover_checkpoints(run_dir: Path) -> list[CheckpointInfo]:
    """List every checkpoint directory under ``run_dir``, oldest first.

    Sort key: epoch when present, else step, else mtime. This is the order
    the retention pruner uses to decide what to delete.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    out: list[CheckpointInfo] = []
    for entry in run_dir.iterdir():
        if not entry.is_dir():
            continue
        epoch, step = _parse_checkpoint_name(entry.name)
        if epoch is None and step is None:
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        out.append(CheckpointInfo(
            name=entry.name,
            path=str(entry.resolve()),
            epoch=epoch,
            step=step,
            size_bytes=_dir_size(entry),
            modified_at=datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc,
            ).isoformat(),
        ))

    def _key(c: CheckpointInfo) -> tuple[int, int, str]:
        return (
            c.epoch if c.epoch is not None else -1,
            c.step if c.step is not None else -1,
            c.modified_at,
        )

    out.sort(key=_key)
    return out


def latest_checkpoint(run_dir: Path) -> CheckpointInfo | None:
    """Most recent checkpoint, or None if there are none."""
    cps = discover_checkpoints(run_dir)
    return cps[-1] if cps else None


def prune_checkpoints(
    run_dir: Path, *, keep_last_n: int,
) -> list[str]:
    """Delete all but the most recent ``keep_last_n`` checkpoint dirs.

    ``keep_last_n == 0`` is a no-op (the user-requested default — keep
    everything). Returns the list of deleted directory names so the caller
    can log / surface it.
    """
    if keep_last_n <= 0:
        return []
    cps = discover_checkpoints(run_dir)
    if len(cps) <= keep_last_n:
        return []
    to_delete = cps[: len(cps) - keep_last_n]
    deleted: list[str] = []
    for cp in to_delete:
        target = Path(cp.path)
        try:
            _rmtree(target)
            deleted.append(cp.name)
        except OSError:
            # Best-effort — log via the runner; do not crash the run.
            pass
    return deleted


def _rmtree(path: Path) -> None:
    """Tolerant rmtree — ignore individual file errors so a partially-locked
    checkpoint dir doesn't take down the whole prune."""
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink(missing_ok=True)
        return
    for child in path.iterdir():
        _rmtree(child)
    try:
        path.rmdir()
    except OSError:
        pass


# ----- run-folder helpers ----------------------------------------------------


def new_run_dir(project: Project, *, label: str | None = None) -> Path:
    """Mint a fresh directory under ``training/runs/`` for a new run.

    Naming: ``YYYYMMDD-HHMMSS[-label]``. The directory is created on disk;
    ``train.py`` is allowed to populate it with epoch checkpoints, eval
    samples, etc.
    """
    runs = project.training_runs_dir
    runs.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{ts}-{label}" if label else ts
    candidate = runs / name
    n = 2
    while candidate.exists():
        candidate = runs / f"{name}-{n}"
        n += 1
    candidate.mkdir(parents=True)
    return candidate


def list_runs(project: Project) -> list[dict]:
    """Return all runs found on disk, newest first, with checkpoint counts."""
    runs = project.training_runs_dir
    if not runs.is_dir():
        return []
    out: list[dict] = []
    for entry in sorted(runs.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        cps = discover_checkpoints(entry)
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            mtime = 0.0
        out.append({
            "name": entry.name,
            "path": str(entry.resolve()),
            "checkpoints": len(cps),
            "latest_checkpoint": cps[-1].name if cps else None,
            "modified_at": datetime.fromtimestamp(
                mtime, tz=timezone.utc,
            ).isoformat() if mtime else "",
        })
    return out


# ----- launcher command ------------------------------------------------------


DEFAULT_LAUNCHER_TEMPLATE = (
    "deepspeed --num_gpus=1 train.py --deepspeed --config {config}"
)


def build_launcher_argv(config: TrainingConfig, *, run_toml: Path) -> list[str]:
    """Build the argv list to launch the trainer.

    The user can override the template via ``launcher_override``. ``{config}``
    is substituted with the run TOML path; if the template lacks the marker,
    we append it as a final positional arg so a bare ``python train.py``
    style override still works.
    """
    template = (config.launcher_override or "").strip() or DEFAULT_LAUNCHER_TEMPLATE
    sub = template.replace("{config}", str(run_toml.resolve()))
    if "{config}" not in template and str(run_toml.resolve()) not in sub:
        sub = f"{sub} {str(run_toml.resolve())}"
    # Naive shell split — good enough for our launcher templates which are
    # space-separated tokens with no embedded quoting.
    return [tok for tok in sub.split() if tok]
