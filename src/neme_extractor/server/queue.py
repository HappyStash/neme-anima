"""Single-worker async job queue with cancellation + event broadcast.

Jobs run **one at a time** so the GPU isn't shared. The queue keeps an internal
ordered list (`_jobs`) and a single asyncio task that pulls from it and runs
the user-supplied `runner` coroutine. Pending jobs can be cancelled instantly;
running jobs receive a cancellation token they should poll periodically.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from neme_extractor.server.events import Broadcaster, Event

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    payload: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    error: str | None = None


# A runner is `async def runner(job_id, payload, broadcaster, cancel_token)`.
Runner = Callable[[str, dict, Broadcaster, asyncio.Event], Awaitable[None]]


class JobQueue:
    """Append-only ordered queue with a single background worker."""

    def __init__(self, *, runner: Runner, broadcaster: Broadcaster | None = None):
        self._runner = runner
        self._broadcaster = broadcaster or Broadcaster()
        self._jobs: list[Job] = []
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._worker_task: asyncio.Task | None = None
        self._stop_requested = False
        self._current_cancel: asyncio.Event | None = None

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._stop_requested = False
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        self._stop_requested = True
        # Tell any running job to wind down — the runner is supposed to poll
        # this token, but even if it doesn't we still want to unblock stop().
        if self._current_cancel is not None:
            self._current_cancel.set()
        self._wake.set()
        if self._worker_task is None:
            return
        task = self._worker_task
        self._worker_task = None
        # Worker should observe _stop_requested on its next loop iteration.
        # If it doesn't (e.g. the runner is wedged in a thread we can't
        # cancel, or any future scheduling oddity), force-cancel after a
        # short grace period so server shutdown can never hang here.
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        except asyncio.CancelledError:
            # If our caller is being cancelled, propagate after still trying
            # to bring the worker down.
            task.cancel()
            raise

    async def submit(self, payload: dict) -> str:
        job_id = secrets.token_hex(6)
        self._jobs.append(Job(job_id=job_id, payload=payload))
        self._idle.clear()
        self._wake.set()
        await self._publish_queue_update()
        return job_id

    async def cancel(self, job_id: str) -> bool:
        for j in self._jobs:
            if j.job_id != job_id:
                continue
            if j.status == JobStatus.PENDING:
                j.status = JobStatus.CANCELLED
                await self._publish_queue_update()
                return True
            if j.status == JobStatus.RUNNING and self._current_cancel is not None:
                self._current_cancel.set()
                return True
        return False

    def snapshot(self) -> list[Job]:
        return [
            Job(job_id=j.job_id, payload=dict(j.payload),
                status=j.status, error=j.error)
            for j in self._jobs
        ]

    async def wait_idle(self) -> None:
        await self._idle.wait()

    @property
    def broadcaster(self) -> Broadcaster:
        return self._broadcaster

    # ----------------- internals -----------------

    async def _worker_loop(self) -> None:
        while True:
            if self._stop_requested:
                return
            job = self._next_pending()
            if job is None:
                self._idle.set()
                self._wake.clear()
                await self._wake.wait()
                continue
            await self._run_one(job)

    def _next_pending(self) -> Job | None:
        for j in self._jobs:
            if j.status == JobStatus.PENDING:
                return j
        return None

    async def _run_one(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        self._current_cancel = asyncio.Event()
        await self._publish_queue_update()
        try:
            await self._runner(job.job_id, job.payload, self._broadcaster, self._current_cancel)
            if self._current_cancel.is_set():
                job.status = JobStatus.CANCELLED
            else:
                job.status = JobStatus.DONE
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            logger.exception("queue.runner failed job_id=%s", job.job_id)
        finally:
            self._current_cancel = None
            await self._publish_queue_update()

    async def _publish_queue_update(self) -> None:
        await self._broadcaster.publish(
            Event(
                type="queue.update",
                payload={"queue": [
                    {
                        "job_id": j.job_id,
                        "status": j.status,
                        "payload": j.payload,
                        "error": j.error,
                    }
                    for j in self._jobs
                ]},
            )
        )
