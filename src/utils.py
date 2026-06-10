import base64
from datetime import datetime

import cv2
import numpy as np

_LLM_MAX_WIDTH = 640
_LLM_MAX_HEIGHT = 360
_JPEG_QUALITY = 85


def encode_frame(frame: np.ndarray) -> str:
    h, w = frame.shape[:2]
    if w > _LLM_MAX_WIDTH or h > _LLM_MAX_HEIGHT:
        scale = min(_LLM_MAX_WIDTH / w, _LLM_MAX_HEIGHT / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    return base64.b64encode(buf).decode()


def format_age(ts: datetime) -> str:
    age = (datetime.now().astimezone() - ts).total_seconds()
    if age < 60:
        return f"{age:.0f}s ago"
    elif age < 3600:
        return f"{age / 60:.0f}m ago"
    else:
        return f"{age / 3600:.1f}h ago"
