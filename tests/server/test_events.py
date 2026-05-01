"""Tests for the WebSocket broadcaster."""

from __future__ import annotations

import asyncio

import pytest

from neme_anima.server.events import Broadcaster, Event


async def test_subscribe_unsubscribe_roundtrip():
    b = Broadcaster()
    q = b.subscribe()
    b.unsubscribe(q)  # must not raise
    assert b.subscriber_count == 0


async def test_publish_fans_out_to_all_subscribers():
    b = Broadcaster()
    q1 = b.subscribe()
    q2 = b.subscribe()
    await b.publish(Event(type="job.progress", payload={"pct": 50}))
    e1 = await asyncio.wait_for(q1.get(), timeout=0.5)
    e2 = await asyncio.wait_for(q2.get(), timeout=0.5)
    assert e1.type == "job.progress" and e1.payload["pct"] == 50
    assert e2.type == "job.progress" and e2.payload["pct"] == 50


async def test_publish_skips_unsubscribed():
    b = Broadcaster()
    q = b.subscribe()
    b.unsubscribe(q)
    await b.publish(Event(type="job.done", payload={}))
    assert q.empty()


async def test_event_to_json_roundtrip():
    e = Event(type="job.frame", payload={"filename": "x.png", "score": 0.9})
    s = e.to_json()
    e2 = Event.from_json(s)
    assert e == e2
