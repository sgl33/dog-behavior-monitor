import base64
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

import cv2
import numpy as np
import requests

from config import Config, WebServerConfig

if TYPE_CHECKING:
    from recorder import Recorder

logger = logging.getLogger(__name__)

_THUMB_W = 320
_THUMB_H = 180
_THUMB_QUALITY = 70


class WebServerClient:
    def __init__(self, config: WebServerConfig):
        self._push_url = config.push_url
        self._public_url = config.public_url

    @property
    def public_url(self) -> str:
        return self._public_url

    def push_result(
        self,
        score: int,
        summary: str,
        description: str,
        ts: datetime,
        frames: list[np.ndarray] | None,
        inference_time: float | None = None,
        cameras: list[str] | None = None,
        detected_by: str | None = None,
        double_pass: bool = False,
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
                    "time": ts.astimezone().isoformat(timespec="seconds"),
                    "score": score,
                    "summary": summary,
                    "description": description,
                    "thumb": thumb,
                    "inference_time": inference_time,
                    "cameras": cameras,
                    "detected_by": detected_by,
                    "double_pass": double_pass,
                },
                timeout=5,
            ).raise_for_status()
        except Exception:
            logger.warning("Failed to push result to web server")

    def push_camera_status(self, statuses: dict[str, dict]) -> None:
        try:
            requests.post(
                self._push_url.replace("/push", "/push_cameras"),
                json={"status": statuses},
                timeout=5,
            ).raise_for_status()
        except Exception:
            logger.warning("Failed to push camera status to web server")

    def run_camera_status_loop(self, recorders: dict[str, "Recorder"], config: Config) -> None:
        """
        Push camera status to web server via HTTP, every second. For each camera
        we report the seconds since its last frame and a state: "ok" (<5s),
        "warn" (5s up to the stale threshold), or "err" (stale / no frames).
        """
        while True:
            time.sleep(1)
            try:
                now = datetime.now()
                statuses = {}
                for cam, rec in recorders.items():
                    ts = rec.last_frame_time()
                    if ts is None:
                        statuses[cam] = {"state": "err", "age": None}
                        continue
                    age = (now - ts).total_seconds()
                    if age >= config.camera_stale_threshold:
                        state = "err"
                    elif age >= 5:
                        state = "warn"
                    else:
                        state = "ok"
                    statuses[cam] = {"state": state, "age": age}
                self.push_camera_status(statuses)
            except Exception:
                logger.exception("Failed to push camera status")
