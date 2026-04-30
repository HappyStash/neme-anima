"""Tests for ``server.job_progress.BroadcasterProgress``."""

from __future__ import annotations

import asyncio

import pytest

from neme_extractor.server.events import Broadcaster, Event
from neme_extractor.server.job_progress import BroadcasterProgress


STAGES = [("setup", "Setup"), ("detect", "Detect"), ("save", "Save")]


async def _drain(sub: asyncio.Queue[Event], *, timeout: float = 0.2) -> list[Event]:
    out: list[Event] = []
    while True:
        try:
            out.append(await asyncio.wait_for(sub.get(), timeout=timeout))
        except asyncio.TimeoutError:
            return out


async def test_initial_snapshot_has_all_stages_pending():
    bc = Broadcaster()
    sub = bc.subscribe()
    p = BroadcasterProgress(
        loop=asyncio.get_running_loop(),
        broadcaster=bc, job_id="j1", project_slug="proj", source_idx=0,
        kind="extract", stages=STAGES,
    )
    p.publish_initial()
    events = await _drain(sub)
    assert len(events) == 1
    payload = events[0].payload
    assert payload["job_id"] == "j1"
    assert payload["source_idx"] == 0
    assert [s["status"] for s in payload["stages"]] == ["pending", "pending", "pending"]


async def test_stage_lifecycle_emits_running_then_done():
    bc = Broadcaster()
    sub = bc.subscribe()
    p = BroadcasterProgress(
        loop=asyncio.get_running_loop(),
        broadcaster=bc, job_id="j1", project_slug="proj", source_idx=0,
        kind="extract", stages=STAGES,
    )
    p.stage_start("setup", "Setup")
    p.stage_done("setup", message="ready")
    p.stage_start("detect", "Detect", total=4)
    for _ in range(4):
        p.stage_advance("detect")
    p.stage_done("detect", message="4 frames")
    p.finish({"kept": 1, "rejected": 0})

    # Allow scheduled coroutines to run.
    await asyncio.sleep(0.05)
    events = await _drain(sub)
    assert len(events) >= 4

    last = events[-1].payload
    assert last["summary"] == {"kept": 1, "rejected": 0}
    by_key = {s["key"]: s for s in last["stages"]}
    assert by_key["setup"]["status"] == "done"
    assert by_key["detect"]["status"] == "done"
    assert by_key["detect"]["pct"] == 1.0
    assert by_key["save"]["status"] == "pending"


async def test_advance_is_throttled():
    bc = Broadcaster()
    sub = bc.subscribe()
    p = BroadcasterProgress(
        loop=asyncio.get_running_loop(),
        broadcaster=bc, job_id="j1", project_slug="proj", source_idx=0,
        kind="extract", stages=STAGES, min_interval_seconds=10.0,
    )
    p.stage_start("detect", "Detect", total=1000)
    for _ in range(1000):
        p.stage_advance("detect")
    p.stage_done("detect")

    await asyncio.sleep(0.05)
    events = await _drain(sub)
    # start (forced) + done (forced) + at most one throttled mid-flight = ≤ 3.
    assert len(events) <= 3
    assert any(e.payload["stages"][1]["status"] == "running" for e in events)
    assert events[-1].payload["stages"][1]["status"] == "done"


async def test_stage_fail_records_error_message():
    bc = Broadcaster()
    sub = bc.subscribe()
    p = BroadcasterProgress(
        loop=asyncio.get_running_loop(),
        broadcaster=bc, job_id="j1", project_slug="proj", source_idx=0,
        kind="extract", stages=STAGES,
    )
    p.stage_fail("detect", "ValueError: oh no")
    await asyncio.sleep(0.02)
    events = await _drain(sub)
    by_key = {s["key"]: s for s in events[-1].payload["stages"]}
    assert by_key["detect"]["status"] == "failed"
    assert "oh no" in by_key["detect"]["message"]
