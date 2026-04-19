"""Sandbox WebSocket endpoint — ``/ws/sandbox/{sid}``.

Receive JSON patches over the wire and stream snapshots back as the
session mutates. Hot-reload path for the frontend's live sandbox.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import api_state

router = APIRouter()


@router.websocket("/ws/sandbox/{sid}")
async def ws(ws: WebSocket, sid: str):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()
            try:
                patch = json.loads(msg)
            except Exception:
                await ws.send_json({"error": "bad json"})
                continue
            sess = api_state.SANDBOX.get(sid)
            if not sess:
                await ws.send_json({"error": "session not found"})
                continue
            sess = api_state.SANDBOX.update(sid, patch)
            await ws.send_json(sess.snapshot())
    except WebSocketDisconnect:
        return
