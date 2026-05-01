"""Async subprocess manager for LoRA training runs.

One :class:`TrainingManager` per server instance handles all projects but
allows at most one active run at a time (training is GPU-bound and would
fight the extraction queue if both ran in parallel). It owns:

* The currently running ``asyncio.subprocess.Process`` (if any).
* A bounded in-memory ring buffer of stdout/stderr lines for the UI's log
  panel (kept small — the on-disk ``run.log`` is the source of truth).
* The persisted run state, so the UI can show "last run was XYZ, stopped at
  epoch N" after a server restart.

Events broadcast on the WebSocket:

* ``training.status`` — payload contains the full status snapshot. Sent on
  every state transition (start, stop, finish) plus periodically while
  training is running.
* ``training.log`` — payload ``{slug, line, stream}`` for one log line.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neme_anima import training as training_lib
from neme_anima.server.events import Broadcaster, Event
from neme_anima.storage.project import Project, TrainingConfig

logger = logging.getLogger(__name__)

# Keep at most this many log lines in-memory for the UI. The on-disk run.log
# captures everything; this buffer is just for fast reconnects.
LOG_BUFFER_MAX = 2000


@dataclass
class RunState:
    """Persisted snapshot for one training run.

    Stored at ``project.training_state_path``; rewritten atomically on each
    state transition. Survives server restarts; on startup we reload it so
    the UI can pick up where it left off.
    """

    project_slug: str
    run_dir: str
    status: str  # "starting" | "running" | "stopping" | "stopped" | "finished" | "failed"
    started_at: str
    finished_at: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    error: str | None = None
    # Best-effort progress parsed from stdout. The trainer prints epoch and
    # step counts; we extract them with simple regexes (see _parse_progress).
    epoch: int | None = None
    step: int | None = None
    loss: float | None = None
    last_log_line: str = ""
    # The checkpoint we resumed from, if any. Useful for the UI to show a
    # "resumed from epoch 20" pill on the run.
    resumed_from: str | None = None
    # Flag that the user clicked Stop. Distinguishes voluntary stop from
    # a crash (status=failed).
    stop_requested: bool = False
    # Snapshot of cfg.epochs at run-launch so the UI can render a progress
    # bar without having to query the live config (which the user may have
    # edited mid-run).
    total_epochs: int | None = None


_PROGRESS_PATTERNS = [
    # diffusion-pipe deepspeed prints lines like:
    #   "epoch: 5, step: 1234, loss: 0.1234"
    # plus tqdm-ish "Epoch 5/40" forms. Extract epoch/step/loss best-effort.
    ("epoch", r"epoch[\s:=]+(\d+)"),
    ("step",  r"\b(?:global[_ ]?step|step)[\s:=]+(\d+)"),
    ("loss",  r"\bloss[\s:=]+([0-9]*\.?[0-9]+)"),
]

import re
_PROG_RE = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in _PROGRESS_PATTERNS]


def _parse_progress(line: str) -> dict[str, Any]:
    """Pull epoch/step/loss out of a log line, if present."""
    out: dict[str, Any] = {}
    for name, rx in _PROG_RE:
        m = rx.search(line)
        if not m:
            continue
        try:
            out[name] = float(m.group(1)) if name == "loss" else int(m.group(1))
        except ValueError:
            pass
    return out


class TrainingManager:
    """Single-active-run training coordinator."""

    def __init__(self, *, broadcaster: Broadcaster) -> None:
        self._broadcaster = broadcaster
        self._lock = asyncio.Lock()
        # The active run, if any. Only one at a time.
        self._project: Project | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._state: RunState | None = None
        self._tasks: list[asyncio.Task] = []
        self._log_buffer: deque[dict[str, Any]] = deque(maxlen=LOG_BUFFER_MAX)
        self._log_path: Path | None = None
        self._log_file: Any = None  # text-mode file handle
        # Tracks the TrainingConfig snapshot at run time so retention pruning
        # uses the correct ``keep_last_n_checkpoints`` even if the user edits
        # config mid-run.
        self._cfg_snapshot: TrainingConfig | None = None

    # ----- public API ------------------------------------------------------

    @property
    def active_slug(self) -> str | None:
        return self._state.project_slug if self._state and self._is_active() else None

    def status(self, project: Project) -> dict:
        """Snapshot for the API/UI. ``running`` is True iff this project owns
        the active subprocess; otherwise we report the last persisted state."""
        is_ours = self._is_active() and self._project is not None and self._project.slug == project.slug
        if is_ours and self._state is not None:
            persisted = self._state
        else:
            persisted = _load_persisted_state(project)
        running = is_ours
        return {
            "slug": project.slug,
            "running": running,
            "global_active_slug": self.active_slug,
            "state": _state_to_dict(persisted) if persisted else None,
            "log_lines": list(self._log_buffer) if is_ours else [],
        }

    def get_log_buffer(self, project: Project) -> list[dict[str, Any]]:
        if self._is_active() and self._project and self._project.slug == project.slug:
            return list(self._log_buffer)
        return []

    async def start(
        self,
        project: Project,
        *,
        resume_from_checkpoint: str | None = None,
        run_dir_name: str | None = None,
    ) -> dict:
        """Kick off a new training run for ``project``.

        Pass ``resume_from_checkpoint`` (a checkpoint *name*) to continue a
        prior run. ``run_dir_name`` reuses an existing run directory; the
        common case is to pass the directory of the run whose checkpoint
        we're resuming from.
        """
        async with self._lock:
            if self._is_active():
                raise RuntimeError(
                    f"another training run is already active "
                    f"(project={self.active_slug})",
                )
            problems = training_lib.validate_for_run(project.training)
            if problems:
                raise RuntimeError("; ".join(problems))

            # Pick / create the run directory.
            project.training_dir.mkdir(parents=True, exist_ok=True)
            project.training_runs_dir.mkdir(parents=True, exist_ok=True)
            if run_dir_name:
                run_dir = project.training_runs_dir / run_dir_name
                run_dir.mkdir(parents=True, exist_ok=True)
            else:
                label = "resume" if resume_from_checkpoint else project.training.preset
                run_dir = training_lib.new_run_dir(project, label=label)

            # Materialize the training dataset as symlinks under the run
            # directory: each kept frame's image points at its crop derivative
            # when one exists, while the sidecar always points at the
            # original `.txt` (so "edit tags on the original; train on the
            # crop" is the on-disk reality the trainer sees). Rebuilt on
            # every start so a resume picks up the latest crop set.
            dataset_dir = run_dir / "dataset"
            staging = training_lib.build_dataset_staging(project, dataset_dir)
            logger.info(
                "training: staged dataset at %s (%d images, %d cropped, %d missing txt)",
                staging["dest"], staging["images"],
                staging["with_crop"], staging["missing_txt"],
            )

            # Render TOML files into the run dir.
            dataset_toml = run_dir / "dataset.toml"
            run_toml = run_dir / "run.toml"
            dataset_toml.write_text(training_lib.render_dataset_toml(
                project, dataset_root=dataset_dir,
            ))
            run_toml.write_text(training_lib.render_run_toml(
                project,
                run_dir=run_dir,
                dataset_toml_path=dataset_toml,
                resume_from_checkpoint=resume_from_checkpoint,
            ))

            # Build argv. Run with cwd=diffusion-pipe-dir so train.py imports
            # its sibling modules correctly.
            argv = training_lib.build_launcher_argv(project.training, run_toml=run_toml)
            cwd = str(Path(project.training.diffusion_pipe_dir).expanduser().resolve())

            self._cfg_snapshot = project.training
            self._project = project
            self._log_buffer.clear()
            self._log_path = run_dir / "run.log"
            self._log_file = open(self._log_path, "a", encoding="utf-8", buffering=1)
            self._log_file.write(
                f"\n=== run start {datetime.now(timezone.utc).isoformat()} "
                f"argv={shlex.join(argv)} cwd={cwd}\n",
            )

            self._state = RunState(
                project_slug=project.slug,
                run_dir=str(run_dir.resolve()),
                status="starting",
                started_at=datetime.now(timezone.utc).isoformat(),
                resumed_from=resume_from_checkpoint,
                total_epochs=project.training.epochs,
            )
            _persist_state(project, self._state)

            try:
                self._proc = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    # New process group so we can signal the whole tree on stop.
                    start_new_session=True,
                )
            except FileNotFoundError as e:
                self._state.status = "failed"
                self._state.error = f"launcher not found: {e}"
                self._state.finished_at = datetime.now(timezone.utc).isoformat()
                _persist_state(project, self._state)
                self._close_log_file()
                self._project = None
                self._cfg_snapshot = None
                raise

            self._state.pid = self._proc.pid
            self._state.status = "running"
            _persist_state(project, self._state)

            # Spawn pump tasks.
            self._tasks = [
                asyncio.create_task(self._pump_stream(self._proc.stdout, "stdout")),
                asyncio.create_task(self._pump_stream(self._proc.stderr, "stderr")),
                asyncio.create_task(self._wait_for_exit()),
            ]

            await self._broadcast_status()
            return self.status(project)

    async def stop(self, project: Project) -> dict:
        """Signal the active run to terminate. Returns the new status."""
        async with self._lock:
            if not self._is_active() or self._project is None or self._project.slug != project.slug:
                raise RuntimeError(
                    "no active training run for this project",
                )
            assert self._proc is not None and self._state is not None
            self._state.stop_requested = True
            self._state.status = "stopping"
            _persist_state(project, self._state)
            await self._broadcast_status()

            # Kill the whole process group — deepspeed spawns child processes.
            try:
                pgid = os.getpgid(self._proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                # Already gone, or process-group lookup failed; fall back.
                try:
                    self._proc.terminate()
                except ProcessLookupError:
                    pass

        # Outside the lock — wait for the wait task to mark things finished.
        await asyncio.sleep(0)  # let scheduler progress
        if self._proc:
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                # Force-kill if it didn't go down cleanly.
                try:
                    pgid = os.getpgid(self._proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    self._proc.kill()
        return self.status(project)

    async def shutdown(self) -> None:
        """Best-effort: kill the active run on server shutdown.

        We don't wait long — the lifespan tear-down should not block the
        user's Ctrl-C. The on-disk state already captures "running with
        pid=X" so the next startup can clean up.
        """
        proc = self._proc
        if proc is None:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        self._close_log_file()

    # ----- internals -------------------------------------------------------

    def _is_active(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def _pump_stream(
        self,
        stream: asyncio.StreamReader | None,
        kind: str,
    ) -> None:
        if stream is None:
            return
        try:
            while True:
                raw = await stream.readline()
                if not raw:
                    break
                try:
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                except Exception:
                    line = repr(raw)
                ts = time.time()
                rec = {"t": ts, "stream": kind, "line": line}
                self._log_buffer.append(rec)
                if self._log_file is not None:
                    try:
                        self._log_file.write(f"[{kind}] {line}\n")
                    except Exception:
                        pass
                # Update progress fields opportunistically.
                if self._state is not None:
                    parsed = _parse_progress(line)
                    if parsed:
                        if "epoch" in parsed:
                            self._state.epoch = parsed["epoch"]
                        if "step" in parsed:
                            self._state.step = parsed["step"]
                        if "loss" in parsed:
                            self._state.loss = parsed["loss"]
                        if self._project is not None:
                            _persist_state(self._project, self._state)
                    self._state.last_log_line = line
                # Push the line to subscribers.
                await self._broadcaster.publish(Event(
                    type="training.log",
                    payload={
                        "slug": self._state.project_slug if self._state else "",
                        "stream": kind,
                        "line": line,
                        "t": ts,
                    },
                ))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("training log pump (%s) crashed", kind)

    async def _wait_for_exit(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            rc = await proc.wait()
        except asyncio.CancelledError:
            raise
        finally:
            # Make sure both stream pumps drain even if wait() returned early.
            for t in self._tasks:
                if t is not asyncio.current_task() and not t.done():
                    try:
                        await asyncio.wait_for(t, timeout=2.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass

        state = self._state
        project = self._project
        if state is not None:
            state.exit_code = rc
            state.finished_at = datetime.now(timezone.utc).isoformat()
            if state.stop_requested:
                state.status = "stopped"
            elif rc == 0:
                state.status = "finished"
            else:
                state.status = "failed"
                if not state.error:
                    state.error = f"trainer exited with code {rc}"
            if project is not None:
                _persist_state(project, state)

        # Apply checkpoint retention now that the run is over.
        if state is not None and self._cfg_snapshot is not None:
            try:
                deleted = training_lib.prune_checkpoints(
                    Path(state.run_dir),
                    keep_last_n=self._cfg_snapshot.keep_last_n_checkpoints,
                )
                if deleted:
                    logger.info(
                        "training: pruned %d checkpoints from %s: %s",
                        len(deleted), state.run_dir, deleted,
                    )
            except Exception:
                logger.exception("training: prune_checkpoints failed")

        await self._broadcast_status()
        self._close_log_file()
        self._proc = None
        self._project = None
        self._cfg_snapshot = None
        self._tasks = []

    def _close_log_file(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
        self._log_file = None

    async def _broadcast_status(self) -> None:
        if self._state is None or self._project is None:
            return
        await self._broadcaster.publish(Event(
            type="training.status",
            payload={
                "slug": self._state.project_slug,
                "running": self._is_active(),
                "state": _state_to_dict(self._state),
            },
        ))


# ----- state persistence helpers --------------------------------------------


def _state_to_dict(state: RunState) -> dict:
    return {
        "project_slug": state.project_slug,
        "run_dir": state.run_dir,
        "run_name": Path(state.run_dir).name,
        "status": state.status,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "pid": state.pid,
        "exit_code": state.exit_code,
        "error": state.error,
        "epoch": state.epoch,
        "step": state.step,
        "loss": state.loss,
        "last_log_line": state.last_log_line,
        "resumed_from": state.resumed_from,
        "stop_requested": state.stop_requested,
        "total_epochs": state.total_epochs,
    }


def _persist_state(project: Project, state: RunState) -> None:
    project.training_dir.mkdir(parents=True, exist_ok=True)
    tmp = project.training_state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_state_to_dict(state), indent=2))
    tmp.replace(project.training_state_path)


def _load_persisted_state(project: Project) -> RunState | None:
    p = project.training_state_path
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    try:
        return RunState(
            project_slug=raw.get("project_slug") or project.slug,
            run_dir=raw["run_dir"],
            status=raw.get("status") or "stopped",
            started_at=raw.get("started_at") or "",
            finished_at=raw.get("finished_at"),
            pid=raw.get("pid"),
            exit_code=raw.get("exit_code"),
            error=raw.get("error"),
            epoch=raw.get("epoch"),
            step=raw.get("step"),
            loss=raw.get("loss"),
            last_log_line=raw.get("last_log_line", ""),
            resumed_from=raw.get("resumed_from"),
            stop_requested=bool(raw.get("stop_requested", False)),
            total_epochs=raw.get("total_epochs"),
        )
    except KeyError:
        return None
