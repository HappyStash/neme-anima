"""WebSocket endpoint at /api/ws.

Each connection subscribes to the broadcaster and forwards every event as a
JSON message until the client disconnects.
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/api", tags=["ws"])


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    broadcaster = websocket.app.state.broadcaster
    sub = broadcaster.subscribe()
    try:
        while True:
            event = await sub.get()
            await websocket.send_text(event.to_json())
    except WebSocketDisconnect:
        pass
    finally:
        broadcaster.unsubscribe(sub)
