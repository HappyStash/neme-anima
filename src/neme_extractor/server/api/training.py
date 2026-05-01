"""REST routes for /api/projects/{slug}/training/*.

The routes are intentionally chatty: separate read/patch on the config,
explicit ``check-path`` for live UI feedback as the user types, and a
``status`` endpoint the frontend polls (or replays via WebSocket) to keep
the run panel current. Heavy work (subprocess management) is delegated to
:class:`neme_extractor.server.training_runner.TrainingManager`.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from neme_extractor import training as training_lib
from neme_extractor.storage.project import Project, TrainingConfig

router = APIRouter(prefix="/api/projects", tags=["training"])


# ----- request bodies --------------------------------------------------------


class TrainingConfigBody(BaseModel):
    # All fields optional so a PATCH can update one knob at a time. Keep the
    # shape in sync with TrainingConfig — fields() drives the assignment loop
    # below to avoid drift.
    preset: str | None = None
    diffusion_pipe_dir: str | None = None
    dit_path: str | None = None
    vae_path: str | None = None
    llm_path: str | None = None
    launcher_override: str | None = None

    rank: int | None = None
    alpha: int | None = None

    learning_rate: float | None = None
    optimizer_betas: list[float] | None = None
    weight_decay: float | None = None
    eps: float | None = None
    warmup_steps: int | None = None
    gradient_clipping: float | None = None

    micro_batch_size: int | None = None
    gradient_accumulation_steps: int | None = None

    resolutions: list[int] | None = None
    enable_ar_bucket: bool | None = None
    min_ar: float | None = None
    max_ar: float | None = None
    num_ar_buckets: int | None = None

    epochs: int | None = None
    eval_every_n_epochs: int | None = None
    save_every_n_epochs: int | None = None

    sigmoid_scale: float | None = None
    llm_adapter_lr: float | None = None

    caption_mode: str | None = None
    tag_dropout_pct: int | None = None
    trigger_token: str | None = None

    keep_last_n_checkpoints: int | None = None


class CheckPathBody(BaseModel):
    path: str
    expect: str = "any"  # "any" | "file" | "dir"


class StartBody(BaseModel):
    resume_from_checkpoint: str | None = None
    run_dir_name: str | None = None


# ----- helpers ---------------------------------------------------------------


def _load_or_404(request: Request, slug: str) -> Project:
    entry = request.app.state.registry.get(slug)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown project: {slug}")
    try:
        return Project.load(Path(entry.folder))
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"project files missing for {slug!r} at {entry.folder}",
        )


def _config_with_path_checks(cfg: TrainingConfig) -> dict:
    """Pair the config dict with a per-path validation dict for the UI."""
    return {
        "config": asdict(cfg),
        "path_checks": {
            "diffusion_pipe_dir": asdict(training_lib.check_path(
                cfg.diffusion_pipe_dir, expect="dir",
            )),
            "dit_path": asdict(training_lib.check_path(
                cfg.dit_path, expect="file",
            )),
            "vae_path": asdict(training_lib.check_path(
                cfg.vae_path, expect="file",
            )),
            "llm_path": asdict(training_lib.check_path(
                cfg.llm_path, expect="file",
            )),
        },
        "problems": training_lib.validate_for_run(cfg),
    }


# ----- config CRUD ----------------------------------------------------------


@router.get("/{slug}/training/config")
async def get_training_config(request: Request, slug: str) -> dict:
    project = _load_or_404(request, slug)
    return _config_with_path_checks(project.training)


@router.patch("/{slug}/training/config")
async def patch_training_config(
    request: Request, slug: str, body: TrainingConfigBody,
) -> dict:
    project = _load_or_404(request, slug)
    payload = body.model_dump(exclude_none=True)
    valid_field_names = {f.name for f in fields(TrainingConfig)}
    for key, value in payload.items():
        if key not in valid_field_names:
            continue
        setattr(project.training, key, value)
    project.save()
    return _config_with_path_checks(project.training)


@router.post("/{slug}/training/check-path")
async def check_path(
    request: Request, slug: str, body: CheckPathBody,
) -> dict:
    # ``slug`` is required to scope the request even though the check is
    # purely filesystem-side; it gives us a simple authz hook later.
    _ = _load_or_404(request, slug)
    return asdict(training_lib.check_path(body.path, expect=body.expect))


# ----- runs / status --------------------------------------------------------


@router.get("/{slug}/training/status")
async def get_status(request: Request, slug: str) -> dict:
    project = _load_or_404(request, slug)
    return request.app.state.training.status(project)


@router.get("/{slug}/training/log")
async def get_log(
    request: Request, slug: str, tail: int = 1000,
) -> dict:
    """Return the last ``tail`` log lines.

    For an active run we read from the in-memory buffer; for a finished run
    we tail the on-disk ``run.log`` of the most recent run directory.
    """
    project = _load_or_404(request, slug)
    mgr = request.app.state.training
    if mgr.active_slug == project.slug:
        return {"source": "live", "lines": mgr.get_log_buffer(project)[-tail:]}
    # Find the latest run dir.
    runs = training_lib.list_runs(project)
    if not runs:
        return {"source": "none", "lines": []}
    run_log = Path(runs[0]["path"]) / "run.log"
    if not run_log.is_file():
        return {"source": "disk", "lines": []}
    try:
        # Cheap tail: read last ~256KB; enough for thousands of lines.
        size = run_log.stat().st_size
        with open(run_log, "rb") as f:
            f.seek(max(0, size - 256 * 1024))
            data = f.read().decode("utf-8", errors="replace")
        lines = data.splitlines()[-tail:]
        return {
            "source": "disk",
            "run_name": runs[0]["name"],
            "lines": [{"line": ln, "stream": "disk", "t": 0} for ln in lines],
        }
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{slug}/training/start", status_code=202)
async def start_training(
    request: Request, slug: str, body: StartBody = StartBody(),
) -> dict:
    project = _load_or_404(request, slug)
    try:
        return await request.app.state.training.start(
            project,
            resume_from_checkpoint=body.resume_from_checkpoint,
            run_dir_name=body.run_dir_name,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/{slug}/training/stop", status_code=202)
async def stop_training(request: Request, slug: str) -> dict:
    project = _load_or_404(request, slug)
    try:
        return await request.app.state.training.stop(project)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/{slug}/training/resume", status_code=202)
async def resume_training(
    request: Request, slug: str, body: StartBody = StartBody(),
) -> dict:
    """Continue a prior run from its last saved DeepSpeed state.

    diffusion-pipe's ``train.py`` resolves ``resume_from_checkpoint`` (string
    form) as a path *relative to* ``output_dir`` and reads
    ``<that>/latest`` for the DeepSpeed checkpoint name. Since each launch
    nests its outputs under a fresh timestamped subdir, the resumable value
    is the *subdir name*, not an ``epoch{N}``-style checkpoint name. We
    pick the most recent subdir that has a ``latest`` marker.

    The run wrapper directory is reused, so the resumed run keeps appending
    checkpoints alongside the prior ones rather than starting a new tree.
    """
    project = _load_or_404(request, slug)
    runs = training_lib.list_runs(project)
    run_name = body.run_dir_name
    resume_target = body.resume_from_checkpoint
    if run_name is None:
        if not runs:
            raise HTTPException(
                status_code=409,
                detail="no prior run found to resume from",
            )
        run_name = runs[0]["name"]
    run_dir = project.training_runs_dir / run_name
    if not run_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"unknown run: {run_name}",
        )
    if resume_target is None:
        resume_target = training_lib.find_resumable_subdir(run_dir)
        if resume_target is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {run_name!r} has no resumable DeepSpeed state — "
                    f"the trainer never finished its first save."
                ),
            )
    # Guardrail: refuse to resume into an empty training window. If the
    # user hasn't raised cfg.epochs since the last save, train.py would
    # still grind through one more epoch (the loop increments before the
    # bound check) and save it — wasting GPU time and bloating disk.
    cps = training_lib.discover_checkpoints(run_dir)
    last_epoch = max(
        (c.epoch for c in cps if c.epoch is not None),
        default=None,
    )
    if last_epoch is not None and project.training.epochs <= last_epoch:
        raise HTTPException(
            status_code=409,
            detail=(
                f"already at epoch {last_epoch} / {project.training.epochs} — "
                f"raise 'epochs' in Settings before resuming."
            ),
        )
    try:
        return await request.app.state.training.start(
            project,
            resume_from_checkpoint=resume_target,
            run_dir_name=run_name,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


# ----- runs + checkpoints ---------------------------------------------------


@router.get("/{slug}/training/runs")
async def list_runs(request: Request, slug: str) -> dict:
    project = _load_or_404(request, slug)
    return {"runs": training_lib.list_runs(project)}


@router.get("/{slug}/training/runs/{run_name}/checkpoints")
async def list_checkpoints(
    request: Request, slug: str, run_name: str,
) -> dict:
    project = _load_or_404(request, slug)
    run_dir = project.training_runs_dir / run_name
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"unknown run: {run_name}")
    cps = training_lib.discover_checkpoints(run_dir)
    return {
        "run_name": run_name,
        "run_dir": str(run_dir.resolve()),
        "checkpoints": [asdict(c) for c in cps],
    }


@router.delete(
    "/{slug}/training/runs/{run_name}/checkpoints/{ckpt_name}",
    status_code=204,
)
async def delete_checkpoint(
    request: Request, slug: str, run_name: str, ckpt_name: str,
    subdir: str = "",
) -> Response:
    """Delete one checkpoint. ``subdir`` is the diffusion-pipe sub-run-dir
    that nests this checkpoint (empty for direct children of ``run_dir``)."""
    project = _load_or_404(request, slug)
    run_dir = project.training_runs_dir / run_name
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"unknown run: {run_name}")
    # Reject any path-injection — both fields are single name components.
    if not ckpt_name or "/" in ckpt_name or ckpt_name in (".", ".."):
        raise HTTPException(status_code=400, detail="invalid checkpoint name")
    if subdir and ("/" in subdir or subdir in (".", "..")):
        raise HTTPException(status_code=400, detail="invalid subdir")
    parent = run_dir / subdir if subdir else run_dir
    target = parent / ckpt_name
    # Belt-and-braces: the resolved target must live under run_dir.
    try:
        target.resolve().relative_to(run_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid checkpoint name")
    if not target.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"unknown checkpoint: {subdir + '/' if subdir else ''}{ckpt_name}",
        )
    training_lib._rmtree(target)
    return Response(status_code=204)


@router.delete("/{slug}/training/runs/{run_name}", status_code=204)
async def delete_run(
    request: Request, slug: str, run_name: str,
) -> Response:
    project = _load_or_404(request, slug)
    if request.app.state.training.active_slug == slug:
        raise HTTPException(
            status_code=409,
            detail="cannot delete a run while training is active",
        )
    run_dir = project.training_runs_dir / run_name
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"unknown run: {run_name}")
    try:
        run_dir.resolve().relative_to(project.training_runs_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid run name")
    training_lib._rmtree(run_dir)
    return Response(status_code=204)


# ----- dataset preview ------------------------------------------------------


@router.get("/{slug}/training/dataset-preview")
async def dataset_preview(request: Request, slug: str) -> dict:
    project = _load_or_404(request, slug)
    return training_lib.dataset_preview(project, sample_n=5)


@router.get("/{slug}/training/run-toml-preview")
async def run_toml_preview(request: Request, slug: str) -> dict:
    """Render the dataset.toml + run.toml the trainer would see.

    Useful for the user to inspect before clicking Start. We use a synthetic
    run-dir path (under training/runs/) without creating it.
    """
    project = _load_or_404(request, slug)
    fake_run = project.training_runs_dir / "<would-be-created-on-start>"
    fake_dataset = fake_run / "dataset.toml"
    return {
        "dataset_toml": training_lib.render_dataset_toml(project),
        "run_toml": training_lib.render_run_toml(
            project,
            run_dir=fake_run,
            dataset_toml_path=fake_dataset,
        ),
        "launcher_argv": training_lib.build_launcher_argv(
            project.training,
            run_toml=fake_run / "run.toml",
        ),
    }
