"""End-to-end WebSocket test using the synchronous TestClient."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from neme_extractor.server.app import create_app
from neme_extractor.server.events import Event


def test_ws_receives_published_events(tmp_path: Path):
    app = create_app(state_dir=tmp_path / "state")
    client = TestClient(app)

    with client.websocket_connect("/api/ws") as ws:
        broadcaster = app.state.broadcaster

        # The TestClient runs the app on its own portal/loop. Schedule a publish
        # via app.state's loop is awkward; use the broadcaster's queue API directly.
        # Push an event onto each subscriber's queue (TestClient already opened
        # the WS, which subscribed).
        async def push():
            await asyncio.sleep(0.05)
            await broadcaster.publish(
                Event(type="job.log", payload={"line": "hello"})
            )

        # Run push() in a fresh loop on a daemon thread so the TestClient's
        # own asyncio loop isn't blocked.
        done = threading.Event()
        def runner():
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(push())
            finally:
                loop.close()
                done.set()
        threading.Thread(target=runner, daemon=True).start()
        done.wait(timeout=2.0)

        msg = ws.receive_json()
        assert msg["type"] == "job.log"
        assert msg["payload"]["line"] == "hello"
