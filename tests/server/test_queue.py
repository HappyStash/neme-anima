"""Tests for the single-worker job queue."""

from __future__ import annotations

import asyncio

from neme_extractor.server.events import Broadcaster
from neme_extractor.server.queue import JobQueue, JobStatus


def _runner_factory(record: list[str], *, fail: bool = False, sleep: float = 0.0):
    async def runner(job_id: str, payload: dict, broadcaster: Broadcaster, cancel_token):
        record.append(f"start:{job_id}")
        if sleep:
            await asyncio.sleep(sleep)
        if cancel_token.is_set():
            record.append(f"cancel:{job_id}")
            return
        if fail:
            raise RuntimeError("boom")
        record.append(f"done:{job_id}")
    return runner


async def test_jobs_run_sequentially():
    record: list[str] = []
    runner = _runner_factory(record, sleep=0.05)
    q = JobQueue(runner=runner)
    await q.start()
    a = await q.submit({"name": "a"})
    b = await q.submit({"name": "b"})
    c = await q.submit({"name": "c"})
    await q.wait_idle()
    await q.stop()
    starts = [r for r in record if r.startswith("start:")]
    assert starts == [f"start:{a}", f"start:{b}", f"start:{c}"]


async def test_cancel_pending_job():
    record: list[str] = []
    runner = _runner_factory(record, sleep=0.10)
    q = JobQueue(runner=runner)
    await q.start()
    running = await q.submit({"name": "running"})
    pending = await q.submit({"name": "pending"})
    # Cancel the pending one before it starts.
    await q.cancel(pending)
    await q.wait_idle()
    await q.stop()
    assert f"start:{running}" in record
    assert f"start:{pending}" not in record


async def test_cancel_running_job_signals_token():
    record: list[str] = []
    async def runner(job_id, payload, broadcaster, cancel_token):
        record.append(f"start:{job_id}")
        for _ in range(20):
            if cancel_token.is_set():
                record.append(f"cancelled:{job_id}")
                return
            await asyncio.sleep(0.02)
        record.append(f"finished:{job_id}")
    q = JobQueue(runner=runner)
    await q.start()
    jid = await q.submit({"name": "x"})
    await asyncio.sleep(0.05)  # let it start
    await q.cancel(jid)
    await q.wait_idle()
    await q.stop()
    assert f"cancelled:{jid}" in record
    assert f"finished:{jid}" not in record


async def test_failed_job_does_not_block_queue():
    record: list[str] = []
    async def runner(job_id, payload, broadcaster, cancel_token):
        record.append(f"start:{job_id}")
        if payload.get("fail"):
            raise RuntimeError("boom")
        record.append(f"done:{job_id}")
    q = JobQueue(runner=runner)
    await q.start()
    await q.submit({"fail": True})
    b = await q.submit({"fail": False})
    await q.wait_idle()
    await q.stop()
    assert f"done:{b}" in record


async def test_status_reflects_pending_running_done():
    runner_event = asyncio.Event()
    async def runner(job_id, payload, broadcaster, cancel_token):
        await runner_event.wait()
    q = JobQueue(runner=runner)
    await q.start()
    a = await q.submit({"name": "a"})
    b = await q.submit({"name": "b"})
    # a should be running, b pending
    await asyncio.sleep(0.02)
    snap = q.snapshot()
    assert snap[0].status == JobStatus.RUNNING and snap[0].job_id == a
    assert snap[1].status == JobStatus.PENDING and snap[1].job_id == b
    runner_event.set()  # let both finish
    await q.wait_idle()
    await q.stop()


async def test_broadcasts_queue_update_on_change():
    broadcaster = Broadcaster()
    sub = broadcaster.subscribe()
    async def runner(job_id, payload, broadcaster, cancel_token):
        return
    q = JobQueue(runner=runner, broadcaster=broadcaster)
    await q.start()
    await q.submit({"name": "x"})
    # Drain a couple of events; we should see at least one queue.update.
    saw_update = False
    for _ in range(5):
        try:
            ev = await asyncio.wait_for(sub.get(), timeout=0.2)
            if ev.type == "queue.update":
                saw_update = True
                break
        except asyncio.TimeoutError:
            break
    await q.wait_idle()
    await q.stop()
    assert saw_update
