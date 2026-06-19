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

# Low-res thumbnails are inlined into every push (and the history replay) so the
# per-message WebSocket payload stays small over slow links (e.g. Tailscale).
# Full-res thumbnails are stored server-side and fetched on demand when the user
# opens a result, so they never bloat the live stream.
_THUMB_LOW_W = 360
_THUMB_LOW_H = 180
_THUMB_HIGH_W = 640
_THUMB_HIGH_H = 360
_THUMB_QUALITY = 70


def _encode_thumb(frame: np.ndarray, max_w: int, max_h: int) -> str:
    """Downscale (never upscale) a frame to fit max_w x max_h and return it as a
    base64-encoded JPEG."""
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _THUMB_QUALITY])
    return base64.b64encode(buf).decode()


class WebServerClient:
    """
    Interface for interacting with the web server for the web app.
    """
    CAMERA_STATUS_PUBLISH_INTERVAL = 0.5

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
        timestamp: datetime,
        frames: list[np.ndarray] | None,
        inference_time: float | None = None,
        cameras: list[str] | None = None,
        detected_by: str | None = None,
        double_pass: bool = False,
    ) -> None:
        """
        Push a single behavioral analysis result to the web server via HTTP.

        Args:
            score (int): Behavior score between 0 and 10.
            summary (str): Short summary of the result.
            description (str): Detailed description of the result.
            timestamp (datetime): Timestamp of the result.
            frames (list[np.ndarray] | None): Optional list of video frames.
                The smallest frame will be used as the thumbnail, and all
                frames will be visible when the user clicks the thumbnail.
            inference_time (float | None): Time taken for inference in seconds.
            cameras (list[str] | None): List of camera names.
            detected_by (str | None): "YOLO" or "LLM"
            double_pass (bool): Whether this result is from a second pass.
        """
        # Encode frames to JPG thumbnails, sorted by image size. Each frame
        # yields a low-res thumbnail (inlined into the push) and a full-res one
        # (stored server-side, fetched on demand when the user opens the result).
        ordered = sorted(frames or [], key=lambda f: f.shape[0] * f.shape[1])
        thumbs = [_encode_thumb(f, _THUMB_LOW_W, _THUMB_LOW_H) for f in ordered]
        full_thumbs = [_encode_thumb(f, _THUMB_HIGH_W, _THUMB_HIGH_H) for f in ordered]

        # Send HTTP request to web server
        try:
            requests.post(
                self._push_url,
                json={
                    "time": timestamp.astimezone().isoformat(timespec="seconds"),
                    "score": score,
                    "summary": summary,
                    "description": description,
                    "thumbs": thumbs,
                    "full_thumbs": full_thumbs,
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
        """
        Push camera status to web server. 

        Args:
            statuses (dict[str, dict]): A dictionary mapping camera name ->
                dict containing the camera's `status` ("ok" | "warn" |"err")
                and `age` (seconds since last frame or None).
        """
        try:
            requests.post(
                self._push_url.replace("/push", "/push_cameras"),
                json={"status": statuses},
                timeout=5,
            ).raise_for_status()
        except Exception:
            logger.warning("Failed to push camera status to web server")

    def run_camera_status_loop(
        self, 
        recorders: dict[str, "Recorder"], 
        config: Config
    ) -> None:
        """
        Push camera status to web server via HTTP periodically. For each camera
        we report the seconds since its last frame and a state: "ok" (<5s),
        "warn" (5s up to the stale threshold), or "err" (stale / no frames).
        """
        while True:
            time.sleep(self.CAMERA_STATUS_PUBLISH_INTERVAL)
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
