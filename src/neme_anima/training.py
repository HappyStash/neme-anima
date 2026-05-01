"""Anima LoRA-training helpers.

This module is **pure plumbing**: it knows how to validate the user's
configuration, render dataset/run TOML files for tdrussell/diffusion-pipe,
discover checkpoints on disk, and prune old ones according to retention
policy. The actual training subprocess lives in
``neme_anima.server.training_runner``.

The TOML output mirrors the reference recipe documented in
``docs/anima-lora-training-notes.md``. We deliberately avoid a TOML library
dependency — the schema is simple, stable, and easier to read in plain
strings than rebuilt through ``tomli_w``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from neme_anima.storage.project import CROP_SUFFIX, Project, TrainingConfig

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
    # Path of the parent directory relative to ``run_dir`` ("" for direct
    # children). diffusion-pipe writes its checkpoints into a timestamped
    # subdirectory under our run wrapper, so this is usually non-empty.
    subdir: str = ""


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

    # Pre-flight the launcher binary so we surface a clear UI message instead
    # of a generic FileNotFoundError out of subprocess_exec at run time.
    token, resolved = resolve_launcher_binary(config)
    if token and resolved is None:
        problems.append(
            f"launcher binary not found: {token!r}. Install diffusion-pipe "
            f"dependencies into a venv at "
            f"'{config.diffusion_pipe_dir or '<diffusion_pipe_dir>'}/.venv' "
            f"(see its README), or set launcher_override to an absolute "
            f"path like '/path/to/env/bin/deepspeed --num_gpus=1 "
            f"train.py --deepspeed --config {{config}}'",
        )

    if not config.resolutions:
        problems.append("at least one training resolution is required")
    if config.epochs <= 0:
        problems.append("epochs must be > 0")
    if config.rank <= 0:
        problems.append("rank must be > 0")
    if config.learning_rate <= 0:
        problems.append("learning_rate must be > 0")
    # Anima's LLM adapter is structurally incompatible with diffusion-pipe's
    # current fp8 path: the adapter's nn.Embedding has ndim==2, so the
    # transformer-dtype cast quantizes its weights to fp8; embedding lookup
    # then returns fp8 tensors that flow into RMSNorm, which crashes
    # promoting fp8*fp32 in the rsqrt branch. Block training before the
    # subprocess gets there so the user sees a UI-actionable message
    # instead of a long deepspeed traceback. Drop this guard if/when
    # diffusion-pipe adds llm_adapter.embed/in_proj/out_proj to its
    # KEEP_IN_HIGH_PRECISION list (see models/cosmos_predict2.py).
    if config.transformer_dtype not in ("", "bfloat16"):
        problems.append(
            f"transformer_dtype={config.transformer_dtype!r} is not "
            "supported for Anima: diffusion-pipe's fp8 path crashes "
            "Anima's LLM adapter (RMSNorm fp8/fp32 promotion error). "
            "Use 'bfloat16' — toggling the 'Fit in 8 GB' preset off and "
            "on resets this field."
        )
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


def render_dataset_toml(project: Project, *, dataset_root: Path | None = None) -> str:
    """Build the dataset.toml that points diffusion-pipe at the training
    dataset directory.

    ``dataset_root`` is the directory diffusion-pipe will scan for image +
    ``.txt`` sidecar pairs. Callers should normally pass the staging
    directory built by :func:`build_dataset_staging`, which substitutes
    each cropped frame's image while keeping the original sidecar. When
    omitted, falls back to ``project.kept_dir`` — useful for previews and
    backwards compatibility, but training will then see both the original
    and the ``_crop`` derivative as separate samples (the historical bug).

    AR bucketing values come from the project's ``TrainingConfig`` so
    ``num_ar_buckets`` and ``min_ar``/``max_ar`` stay in sync with the run
    config.
    """
    cfg = project.training
    root = dataset_root if dataset_root is not None else project.kept_dir
    lines = [
        "# Auto-generated by neme-anima — do not hand-edit.",
        f'# Project: {project.slug}',
        "",
        "resolutions = " + _toml_value(cfg.resolutions),
        _toml_kv("enable_ar_bucket", cfg.enable_ar_bucket),
        _toml_kv("min_ar", cfg.min_ar),
        _toml_kv("max_ar", cfg.max_ar),
        _toml_kv("num_ar_buckets", cfg.num_ar_buckets),
        "",
        "[[directory]]",
        _toml_kv("path", str(Path(root).resolve())),
        _toml_kv("num_repeats", 1),
        "",
    ]
    return "\n".join(lines)


# ----- training dataset staging ---------------------------------------------


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def build_dataset_staging(project: Project, dest: Path) -> dict:
    """Materialize the trainer's view of the project as a directory of
    symlinks under ``dest``.

    For each kept frame ``<stem>``:
      * the staged image ``<dest>/<stem>.<ext>`` points at
        ``<stem>_crop.png`` if a crop derivative exists, else at
        ``<stem>.<ext>``. The crop's pixels are what the model should see.
      * the staged sidecar ``<dest>/<stem>.txt`` always points at the
        original ``<stem>.txt`` — there is only ever one .txt per kept
        frame, regardless of cropping. This is what makes "edit tags on
        the original; train on the crop" work end-to-end.

    ``_crop`` images in ``kept_dir`` are skipped as standalone samples;
    they are only used as the link target for their parent. Any legacy
    ``<stem>_crop.txt`` files on disk are ignored.

    The destination is wiped and recreated on every call so a re-run picks
    up the latest crop set without dragging stale links forward.
    """
    kept = project.kept_dir
    dest = Path(dest)
    if dest.exists():
        for child in dest.iterdir():
            try:
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            except OSError:
                pass
    dest.mkdir(parents=True, exist_ok=True)

    paired = 0
    with_crop = 0
    missing_txt = 0
    if not kept.is_dir():
        return {
            "images": 0, "with_crop": 0, "missing_txt": 0,
            "dest": str(dest.resolve()),
        }

    for img in sorted(kept.iterdir()):
        if not img.is_file():
            continue
        if img.suffix.lower() not in _IMAGE_EXTS:
            continue
        stem = img.stem
        if stem.endswith(CROP_SUFFIX):
            continue  # handled via its parent
        crop = kept / f"{stem}{CROP_SUFFIX}.png"
        src_img = crop if crop.is_file() else img
        # Stage the image under the original stem so the trainer pairs it
        # with the original sidecar by stem-match.
        link_img = dest / f"{stem}{src_img.suffix.lower()}"
        _link_or_copy(src_img, link_img)
        src_txt = kept / f"{stem}.txt"
        if src_txt.is_file():
            _link_or_copy(src_txt, dest / f"{stem}.txt")
        else:
            missing_txt += 1
        paired += 1
        if crop.is_file():
            with_crop += 1
    return {
        "images": paired,
        "with_crop": with_crop,
        "missing_txt": missing_txt,
        "dest": str(dest.resolve()),
    }


def _link_or_copy(src: Path, dest: Path) -> None:
    """Symlink ``dest`` → ``src`` (absolute target); fall back to a copy
    on platforms or filesystems that reject symlinks (e.g. Windows without
    developer mode)."""
    src_abs = src.resolve()
    try:
        os.symlink(src_abs, dest)
    except OSError:
        shutil.copy2(src_abs, dest)


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
        "# Auto-generated by neme-anima — do not hand-edit.",
        f"# Project: {project.slug}",
        f"# Generated at {datetime.now(timezone.utc).isoformat()}",
        "",
        _toml_kv("output_dir", str(run_dir.resolve())),
        _toml_kv("dataset", str(dataset_toml_path.resolve())),
    ]
    # ``resume_from_checkpoint`` MUST be a top-level key in the TOML — if
    # we let it fall after a ``[section]`` header, train.py would read it
    # off the wrong table (and DeepSpeed forwarded it as a kwarg to AdamW,
    # blowing up with TypeError).
    if resume_from_checkpoint:
        lines.append(
            _toml_kv("resume_from_checkpoint", resume_from_checkpoint),
        )
    # ``blocks_to_swap`` is a top-level diffusion-pipe option — like
    # ``resume_from_checkpoint`` it must not slip under a [section] header
    # or train.py won't see it.
    if cfg.blocks_to_swap > 0:
        lines.append(_toml_kv("blocks_to_swap", cfg.blocks_to_swap))
    lines += [
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
        # ``activation_checkpointing`` accepts either ``true`` (PyTorch
        # native checkpoint, the recipe default) or the string ``"unsloth"``
        # (unsloth's variant, more aggressive memory savings — used by the
        # canonical wan_14b_min_vram example). diffusion-pipe's train.py
        # branches on ``== True`` vs ``== 'unsloth'``, so we must emit a
        # bare boolean for the default case, not the string ``"true"``.
        _toml_kv(
            "activation_checkpointing",
            "unsloth" if cfg.activation_checkpointing_mode == "unsloth" else True,
        ),
        _toml_kv("partition_method", "parameters"),
        _toml_kv("save_dtype", "bfloat16"),
        _toml_kv("caching_batch_size", 1),
        _toml_kv("map_num_proc", 8),
        _toml_kv("steps_per_print", 1),
        # torch.compile is incompatible with the low-VRAM levers: Inductor
        # fails to lower fused kernels that touch fp8 weights (Triton MLIR
        # PassManager error on embedding+RMSNorm fusions), and block_swap
        # moves params between CPU/GPU mid-step, breaking compile's
        # static-parameter assumption. The canonical wan_14b_min_vram.toml
        # / qwen_image_24gb_vram.toml examples shipped with diffusion-pipe
        # omit ``compile`` for the same reason.
        *(
            [_toml_kv("compile", True)]
            if cfg.transformer_dtype == "bfloat16" and cfg.blocks_to_swap == 0
            else []
        ),
        "",
        "[model]",
        _toml_kv("type", "anima"),
        _toml_kv("transformer_path", str(Path(cfg.dit_path).expanduser().resolve())),
        _toml_kv("vae_path", str(Path(cfg.vae_path).expanduser().resolve())),
        _toml_kv("llm_path", str(Path(cfg.llm_path).expanduser().resolve())),
        # ``dtype`` is the base storage/compute dtype — must be a real fp
        # type. The VAE init does ``1.0 / std`` on this dtype, which fails
        # on float8 (NotImplementedError "reciprocal_cpu" for Float8_e4m3fn).
        # Keep it pinned to bfloat16; the optional fp8 quantization layer
        # is a separate ``transformer_dtype`` key applied on top.
        _toml_kv("dtype", "bfloat16"),
        # Only emit transformer_dtype when it differs from the base — that
        # keeps default-recipe TOMLs unchanged and matches diffusion-pipe's
        # own ``model_config.get('transformer_dtype', dtype)`` fallback.
        *(
            [_toml_kv("transformer_dtype", cfg.transformer_dtype)]
            if cfg.transformer_dtype and cfg.transformer_dtype != "bfloat16"
            else []
        ),
        _toml_kv("llm_adapter_lr", cfg.llm_adapter_lr),
        _toml_kv("sigmoid_scale", cfg.sigmoid_scale),
        "",
        "[adapter]",
        _toml_kv("type", "lora"),
        _toml_kv("rank", cfg.rank),
        _toml_kv("dtype", "bfloat16"),
        "",
        "[optimizer]",
        # ``optimizer_type`` is forwarded as the diffusion-pipe ``type`` key.
        # train.py lowercases it before matching, so casing is free; we
        # preserve user casing for readability ("AdamW8bitKahan" vs
        # "adamw_optimi"). Optimizer kwargs (lr / betas / weight_decay /
        # eps) flow through to all supported types unchanged — see
        # train.py's ``kwargs = {k: v for k, v in optim_config.items()
        # if k not in ['type', 'gradient_release']}``.
        _toml_kv("type", cfg.optimizer_type),
        _toml_kv("lr", cfg.learning_rate),
        _toml_kv("betas", cfg.optimizer_betas),
        _toml_kv("weight_decay", cfg.weight_decay),
        _toml_kv("eps", cfg.eps),
        "",
    ]
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
    # `_crop` derivatives are internal training-target replacements, not
    # standalone samples — counting them would double the visible total
    # whenever the user has cropped a frame.
    images = sorted(
        p for p in kept.iterdir()
        if p.is_file()
        and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        and not p.stem.endswith(CROP_SUFFIX)
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

    diffusion-pipe creates a per-launch timestamped subdirectory inside
    ``run_dir`` and writes its ``epoch{N}`` / ``global_step{N}`` checkpoints
    *there*, not directly in ``run_dir``. We therefore walk both levels: a
    checkpoint can live as a direct child of ``run_dir`` (legacy / unusual
    layouts) or one level deeper (the common case).

    Sort key: epoch when present, else step, else mtime. This is the order
    the retention pruner uses to decide what to delete.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return []
    out: list[CheckpointInfo] = []

    def _record(entry: Path, rel_subdir: str) -> None:
        epoch, step = _parse_checkpoint_name(entry.name)
        if epoch is None and step is None:
            return
        try:
            stat = entry.stat()
        except OSError:
            return
        out.append(CheckpointInfo(
            name=entry.name,
            path=str(entry.resolve()),
            epoch=epoch,
            step=step,
            size_bytes=_dir_size(entry),
            modified_at=datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc,
            ).isoformat(),
            subdir=rel_subdir,
        ))

    # Skip directories we know aren't checkpoints (the staged training
    # dataset; we'd just waste a sysstat scan walking it).
    SKIP = {"dataset"}
    for entry in run_dir.iterdir():
        if not entry.is_dir() or entry.name in SKIP:
            continue
        if _parse_checkpoint_name(entry.name) != (None, None):
            _record(entry, "")
            continue
        # Treat anything else as a candidate diffusion-pipe sub-run-dir.
        try:
            for inner in entry.iterdir():
                if inner.is_dir():
                    _record(inner, entry.name)
        except OSError:
            continue

    def _key(c: CheckpointInfo) -> tuple[int, int, str]:
        return (
            c.epoch if c.epoch is not None else -1,
            c.step if c.step is not None else -1,
            c.modified_at,
        )

    out.sort(key=_key)
    return out


def find_resumable_subdir(run_dir: Path) -> str | None:
    """Return the diffusion-pipe sub-run-dir name we can resume from.

    train.py reads ``<output_dir>/<resume_from_checkpoint>/latest`` to find
    its DeepSpeed state, so the value to pass back to ``train.py`` is *not*
    a checkpoint name but the timestamped subdirectory created on a prior
    launch. We pick the most recent subdir that has a ``latest`` file.
    Returns ``None`` if nothing resumable exists.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    candidates: list[tuple[float, str]] = []
    for entry in run_dir.iterdir():
        if not entry.is_dir() or entry.name == "dataset":
            continue
        latest = entry / "latest"
        if not latest.is_file():
            continue
        try:
            mtime = latest.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, entry.name))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


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
        # Split LoRA outputs from DeepSpeed plumbing — the UI shows the
        # former as artifacts, while the latter only matters for resume.
        epoch_cps = [c for c in cps if c.epoch is not None]
        latest_epoch = max(
            (c.epoch for c in epoch_cps if c.epoch is not None),
            default=None,
        )
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            mtime = 0.0
        out.append({
            "name": entry.name,
            "path": str(entry.resolve()),
            "checkpoints": len(epoch_cps),
            "latest_checkpoint": epoch_cps[-1].name if epoch_cps else None,
            "latest_epoch": latest_epoch,
            "resumable_subdir": find_resumable_subdir(entry),
            "total_size_bytes": sum(c.size_bytes for c in cps),
            "modified_at": datetime.fromtimestamp(
                mtime, tz=timezone.utc,
            ).isoformat() if mtime else "",
        })
    return out


# ----- launcher command ------------------------------------------------------


DEFAULT_LAUNCHER_TEMPLATE = (
    "deepspeed --num_gpus=1 train.py --deepspeed --config {config}"
)


def resolve_launcher_binary(config: TrainingConfig) -> tuple[str, str | None]:
    """Find the absolute path of the launcher's first token.

    Returns ``(token, resolved_path_or_None)``. We look in the system PATH
    first, then in ``<diffusion_pipe_dir>/.venv/bin`` and ``venv/bin`` so a
    user who configured the diffusion-pipe directory and built its venv
    there doesn't also have to set ``launcher_override`` — the natural
    install layout just works.
    """
    template = (config.launcher_override or "").strip() or DEFAULT_LAUNCHER_TEMPLATE
    token = next((tok for tok in template.split() if tok), "")
    if not token:
        return ("", None)
    found = shutil.which(token)
    if found is None and config.diffusion_pipe_dir:
        dp = Path(config.diffusion_pipe_dir).expanduser()
        for venv_bin in (dp / ".venv" / "bin", dp / "venv" / "bin"):
            cand = venv_bin / token
            if cand.is_file() and os.access(cand, os.X_OK):
                found = str(cand)
                break
    return (token, found)


def _shim_path() -> Path:
    """Absolute path to the diffusion-pipe pre-train shim shipped with this
    package. The shim patches diffusion-pipe's offloader to use blocking
    GPU↔CPU transfers — the workaround for WSL2's pinned-memory ceiling.
    See the shim file's docstring for the full root-cause analysis."""
    return Path(__file__).parent / "_diffusion_pipe_shim.py"


def build_launcher_argv(config: TrainingConfig, *, run_toml: Path) -> list[str]:
    """Build the argv list to launch the trainer.

    The user can override the template via ``launcher_override``. ``{config}``
    is substituted with the run TOML path; if the template lacks the marker,
    we append it as a final positional arg so a bare ``python train.py``
    style override still works.

    The launcher binary's first token is resolved to an absolute path when
    possible so the spawned subprocess finds it regardless of the parent
    process's PATH (e.g. the server is started without the diffusion-pipe
    venv activated).

    When ``blocks_to_swap > 0``, the trainer script (``train.py``) is
    transparently replaced with our pre-train shim. The shim monkey-patches
    diffusion-pipe's offloader to use blocking GPU↔CPU transfers (which
    don't need pinned host memory) before exec'ing train.py — fixing the
    misleading "CUDA out of memory" error that hits WSL2 users with plenty
    of free VRAM. The substitution is keyed on the bare token "train.py"
    so user-supplied launcher_override templates that name train.py
    explicitly are still upgraded; templates that point at a custom
    trainer script are left alone (the user knows what they're doing).
    """
    template = (config.launcher_override or "").strip() or DEFAULT_LAUNCHER_TEMPLATE
    sub = template.replace("{config}", str(run_toml.resolve()))
    if "{config}" not in template and str(run_toml.resolve()) not in sub:
        sub = f"{sub} {str(run_toml.resolve())}"
    # Naive shell split — good enough for our launcher templates which are
    # space-separated tokens with no embedded quoting.
    argv = [tok for tok in sub.split() if tok]
    if argv:
        _, resolved = resolve_launcher_binary(config)
        if resolved:
            argv[0] = resolved
        # Swap train.py for the shim when block-swap is on. The shim is
        # invoked by deepspeed exactly like train.py; argv[1:] (--deepspeed
        # --config <toml>) is forwarded unchanged because the shim exec's
        # train.py with the same sys.argv.
        if config.blocks_to_swap > 0:
            shim = str(_shim_path().resolve())
            argv = [shim if tok == "train.py" else tok for tok in argv]
    return argv
