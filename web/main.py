import json
import logging
from collections import deque

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

_history: deque = deque(maxlen=100)
_clients: set[WebSocket] = set()
_camera_status: dict[str, bool] = {}


class PushPayload(BaseModel):
    time: str
    score: int
    summary: str | None = None
    description: str
    thumb: str | None = None
    inference_time: float | None = None
    cameras: list[str] | None = None
    detected_by: str | None = None


class CameraStatusPayload(BaseModel):
    status: dict[str, bool]


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/push")
async def push(payload: PushPayload):
    entry = payload.model_dump()
    _history.append(entry)
    await _broadcast({"type": "result", "entry": entry})
    return JSONResponse({"ok": True})


@app.post("/push_cameras")
async def push_cameras(payload: CameraStatusPayload):
    global _camera_status
    _camera_status = payload.status
    await _broadcast({"type": "cameras", "status": _camera_status})
    return JSONResponse({"ok": True})


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _clients.add(websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "history",
            "entries": list(_history),
        }))
        if _camera_status:
            await websocket.send_text(json.dumps({
                "type": "cameras",
                "status": _camera_status,
            }))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(websocket)


async def _broadcast(msg: dict) -> None:
    dead: set[WebSocket] = set()
    text = json.dumps(msg)
    for ws in list(_clients):
        try:
            await ws.send_text(text)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)
