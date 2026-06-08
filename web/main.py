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


class PushPayload(BaseModel):
    time: str
    score: int
    description: str
    thumb: str | None = None


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/push")
async def push(payload: PushPayload):
    entry = payload.model_dump()
    _history.append(entry)
    await _broadcast(entry)
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
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(websocket)


async def _broadcast(entry: dict) -> None:
    dead: set[WebSocket] = set()
    msg = json.dumps({"type": "result", "entry": entry})
    for ws in list(_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)
