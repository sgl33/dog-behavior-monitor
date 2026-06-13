import base64
import logging
import os
import subprocess
import tempfile
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger(__name__)

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


VIDEO_SIZE = (960, 540)


def compile_video(frames: list[np.ndarray], fps: float) -> bytes:
    if not frames:
        raise ValueError("No frames to compile into video")
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, frame in enumerate(frames):
            resized = cv2.resize(frame, VIDEO_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(tmp_dir, f"f{i:04d}.jpg"), resized)
        out = os.path.join(tmp_dir, "out.mp4")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-framerate", str(int(fps)),
                    "-i", os.path.join(tmp_dir, "f%04d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    out,
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("ffmpeg failed (rc=%d): %s", e.returncode, e.stderr.decode(errors="replace"))
            raise
        with open(out, "rb") as f:
            return f.read()


def format_age(ts: datetime) -> str:
    age = (datetime.now().astimezone() - ts).total_seconds()
    if age < 60:
        return f"{age:.0f}s ago"
    elif age < 3600:
        return f"{age / 60:.0f}m ago"
    else:
        return f"{age / 3600:.1f}h ago"
