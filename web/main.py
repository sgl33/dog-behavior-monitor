import asyncio
import json
import logging
from collections import OrderedDict, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Periodic heartbeat so connected clients can tell a live connection from a
# stale one. Without it, a half-open socket (laptop sleep, network drop) keeps
# looking "live" in the browser because no data and no close event arrive.
HEARTBEAT_INTERVAL = 5


async def _heartbeat_loop() -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        await _broadcast({"type": "ping"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_heartbeat_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

_history: deque = deque(maxlen=50)
_clients: set[WebSocket] = set()
_camera_status: dict[str, dict] = {}
# Full-res thumbnails kept out of the live stream and fetched on demand, keyed
# by entry id. Bounded so memory tracks the (capped) history.
_full_thumbs: "OrderedDict[str, list[str]]" = OrderedDict()
_FULL_THUMBS_MAX = 50
_next_id = 0


class PushPayload(BaseModel):
    time: str
    score: int
    summary: str | None = None
    description: str
    thumbs: list[str] | None = None
    full_thumbs: list[str] | None = None
    inference_time: float | None = None
    cameras: list[str] | None = None
    detected_by: str | None = None
    double_pass: bool = False


class CameraInfo(BaseModel):
    state: str  # "ok" | "warn" | "err"
    age: float | None = None  # seconds since last frame, None if no frames yet


class CameraStatusPayload(BaseModel):
    status: dict[str, CameraInfo]


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/push")
async def push(payload: PushPayload):
    global _next_id
    entry = payload.model_dump()
    # Pull the full-res thumbnails out of the broadcast/history payload and
    # stash them separately under a fresh id; the client fetches them lazily.
    full = entry.pop("full_thumbs", None)
    entry_id = str(_next_id)
    _next_id += 1
    entry["id"] = entry_id
    if full:
        _full_thumbs[entry_id] = full
        while len(_full_thumbs) > _FULL_THUMBS_MAX:
            _full_thumbs.popitem(last=False)
    _history.append(entry)
    await _broadcast({"type": "result", "entry": entry})
    return JSONResponse({"ok": True})


@app.get("/thumbs/{entry_id}")
async def thumbs(entry_id: str):
    full = _full_thumbs.get(entry_id)
    if full is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"thumbs": full})


@app.post("/push_cameras")
async def push_cameras(payload: CameraStatusPayload):
    global _camera_status
    _camera_status = {name: info.model_dump() for name, info in payload.status.items()}
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
