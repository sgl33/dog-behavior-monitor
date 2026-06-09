import base64
import logging
from datetime import datetime

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

_THUMB_W = 320
_THUMB_H = 180
_THUMB_QUALITY = 70


class WebServerClient:
    def __init__(self, push_url: str, public_url: str):
        self._push_url = push_url
        self._public_url = public_url

    @property
    def public_url(self) -> str:
        return self._public_url

    def push_result(
        self,
        score: int,
        description: str,
        ts: datetime,
        frames: list[np.ndarray] | None,
        inference_time: float | None = None,
    ) -> None:
        thumb = None
        if frames:
            frame = frames[len(frames) // 2]
            h, w = frame.shape[:2]
            scale = min(_THUMB_W / w, _THUMB_H / h, 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _THUMB_QUALITY])
            thumb = base64.b64encode(buf).decode()

        try:
            requests.post(
                self._push_url,
                json={
                    "time": ts.strftime("%H:%M:%S"),
                    "score": score,
                    "description": description,
                    "thumb": thumb,
                    "inference_time": inference_time,
                },
                timeout=5,
            ).raise_for_status()
        except Exception:
            logger.warning("Failed to push result to web server")
