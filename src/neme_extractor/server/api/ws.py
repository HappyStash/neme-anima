"""WebSocket endpoint at /api/ws.

Each connection subscribes to the broadcaster and forwards every event as a
JSON message until the client disconnects.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/api", tags=["ws"])


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    broadcaster = websocket.app.state.broadcaster
    sub = broadcaster.subscribe()

    # Race a receive task (which raises WebSocketDisconnect on close) against
    # broadcaster.get(). Without this, a closed client leaves the handler
    # parked in `sub.get()` forever, and uvicorn's "Waiting for background
    # tasks to complete" never finishes on Ctrl+C.
    recv_task = asyncio.create_task(websocket.receive_text())
    try:
        while True:
            send_task = asyncio.create_task(sub.get())
            done, _ = await asyncio.wait(
                {recv_task, send_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if recv_task in done:
                send_task.cancel()
                # Either the client sent a message (we ignore — this channel
                # is server-push only) or it disconnected. receive_text()
                # raises WebSocketDisconnect on close.
                exc = recv_task.exception()
                if exc is not None:
                    raise exc
                recv_task = asyncio.create_task(websocket.receive_text())
                continue
            event = send_task.result()
            await websocket.send_text(event.to_json())
    except WebSocketDisconnect:
        pass
    finally:
        recv_task.cancel()
        broadcaster.unsubscribe(sub)
