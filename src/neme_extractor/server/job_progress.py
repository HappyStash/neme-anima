"""Broadcaster-backed implementation of :class:`PipelineProgress`.

The pipeline runs synchronously on a worker thread (see
``server.app._pipeline_runner``); WebSocket fan-out is async on the event
loop. ``BroadcasterProgress`` bridges the two: stage callbacks are invoked
from the worker, state is mutated under a lock, and a debounced publish task
is scheduled on the captured event loop via ``run_coroutine_threadsafe``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from neme_extractor.pipeline_progress import PipelineProgress
from neme_extractor.server.events import Broadcaster, Event


class BroadcasterProgress(PipelineProgress):
    """Per-job progress reporter that publishes ``job.stages`` events.

    Construct with the asyncio loop the broadcaster lives on, then hand the
    instance to ``run_extract`` / ``run_rerun``. Every state mutation is
    coalesced — at most one event per ``min_interval_seconds`` plus one
    immediate event on stage transitions (start / done / fail / finish).
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        broadcaster: Broadcaster,
        job_id: str,
        project_slug: str,
        source_idx: int | None,
        kind: str,
        stages: list[tuple[str, str]],
        min_interval_seconds: float = 0.15,
    ) -> None:
        self._loop = loop
        self._broadcaster = broadcaster
        self._job_id = job_id
        self._project = project_slug
        self._source_idx = source_idx
        self._kind = kind
        self._min_interval = min_interval_seconds
        self._lock = threading.Lock()
        self._stages: dict[str, dict[str, Any]] = {
            key: {
                "key": key, "label": label, "status": "pending",
                "current": 0, "total": 0, "pct": 0.0, "message": "",
            }
            for key, label in stages
        }
        self._stage_order: list[str] = [k for k, _ in stages]
        self._summary: dict[str, Any] | None = None
        self._last_publish: float = 0.0
        self._published_done: bool = False
        # Pause/resume — used by the pipeline to wait for a user action (e.g.
        # reviewing kept frames before tagging). Set from the asyncio side
        # via ``resume()``.
        self._pause_event = threading.Event()
        self._paused: bool = False
        self._pause_message: str = ""

    # ----------- PipelineProgress callbacks (called from worker thread) -----

    def stage_start(self, key: str, label: str, *, total: int = 0, message: str = "") -> None:
        with self._lock:
            s = self._stages.get(key)
            if s is None:
                return
            s["label"] = label or s["label"]
            s["status"] = "running"
            s["total"] = int(total)
            s["current"] = 0
            s["pct"] = 0.0
            s["message"] = message
        self._publish(force=True)

    def stage_advance(self, key: str, n: int = 1) -> None:
        with self._lock:
            s = self._stages.get(key)
            if s is None:
                return
            s["current"] += int(n)
            if s["total"] > 0:
                s["pct"] = max(0.0, min(1.0, s["current"] / s["total"]))
        self._publish(force=False)

    def stage_message(self, key: str, message: str) -> None:
        with self._lock:
            s = self._stages.get(key)
            if s is None:
                return
            s["message"] = message
        self._publish(force=False)

    def stage_done(self, key: str, *, message: str = "") -> None:
        with self._lock:
            s = self._stages.get(key)
            if s is None:
                return
            s["status"] = "done"
            s["pct"] = 1.0
            if s["total"] and s["current"] < s["total"]:
                s["current"] = s["total"]
            if message:
                s["message"] = message
        self._publish(force=True)

    def stage_fail(self, key: str, error: str) -> None:
        with self._lock:
            # Prefer the currently-running stage so the red ✕ lands in the
            # right place even when the caller passes a generic key like
            # "setup". Falls back to the named key, then to the first
            # not-yet-done stage.
            running = next(
                (k for k in self._stage_order if self._stages[k]["status"] == "running"),
                None,
            )
            target = (
                running
                or (key if key in self._stages and self._stages[key]["status"] != "done" else None)
                or self._first_running_or_pending()
                or key
            )
            if target not in self._stages:
                return
            s = self._stages[target]
            s["status"] = "failed"
            s["message"] = error[:300]
        self._publish(force=True)

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._summary = dict(summary) if summary else None
        self._publish(force=True)

    # ----------- internals --------------------------------------------------

    def _first_running_or_pending(self) -> str | None:
        for key in self._stage_order:
            st = self._stages[key]["status"]
            if st in ("running", "pending"):
                return key
        return None

    def _snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self._job_id,
            "project": self._project,
            "source_idx": self._source_idx,
            "kind": self._kind,
            "stages": [self._stages[k].copy() for k in self._stage_order],
            "summary": dict(self._summary) if self._summary is not None else None,
            "paused": self._paused,
            "pause_message": self._pause_message,
        }

    # ----------- pause / resume -------------------------------------------

    def wait_for_resume(self, *, message: str = "") -> None:
        """Called from the worker thread. Marks the job as paused, publishes
        an immediate update so the UI flips to the pause indicator, then
        blocks until ``resume()`` is called from the asyncio side.
        """
        with self._lock:
            self._paused = True
            self._pause_message = message
            self._pause_event.clear()
        self._publish(force=True)
        self._pause_event.wait()
        with self._lock:
            self._paused = False
            self._pause_message = ""
        self._publish(force=True)

    def resume(self) -> None:
        """Called from the asyncio side (API handler) to release a paused job."""
        self._pause_event.set()

    @property
    def is_paused(self) -> bool:
        return self._paused

    def _publish(self, *, force: bool) -> None:
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._last_publish) < self._min_interval:
                return
            self._last_publish = now
            payload = self._snapshot()
        event = Event(type="job.stages", payload=payload)
        try:
            asyncio.run_coroutine_threadsafe(self._broadcaster.publish(event), self._loop)
        except RuntimeError:
            # Loop closed — ignore (we're shutting down).
            pass

    def publish_initial(self) -> None:
        """Publish the initial all-pending snapshot (called from the loop thread)."""
        payload = self._snapshot()
        # Already on the event loop, so just await directly via task.
        asyncio.create_task(self._broadcaster.publish(Event(type="job.stages", payload=payload)))
