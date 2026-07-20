import base64
import json
import logging
import os
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_LEVEL_COLORS = {
    logging.DEBUG:    "\033[90m",   # gray
    logging.INFO:     "\033[97m",   # white
    logging.WARNING:  "\033[33m",   # yellow
    logging.ERROR:    "\033[31m",   # red
    logging.CRITICAL: "\033[35m",   # magenta
}
_RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        # strftime has no sub-second support, so append tenths of a second.
        return f"{super().formatTime(record, datefmt)}.{int(record.msecs // 100)}"

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname:<8}{_RESET}"
        return super().format(record)

_LLM_MAX_WIDTH = 640
_LLM_MAX_HEIGHT = 360
_JPEG_QUALITY = 85
VIDEO_SIZE = (960, 540)


def select_boxes(
    scored: list[tuple[tuple[int, int, int, int], float]],
    selection: str,
) -> list[tuple[int, int, int, int]]:
    """
    Reduce one frame's scored dog boxes to the region the crop should span.

    `scored` pairs each box with the YOLO confidence to rank it by.

    "highest" keeps only the best-scoring box. "union" keeps them all, and
    `_crop` then spans them with a single bounding box, so one false positive
    across the room widens the crop to cover both it and the dog.
    """
    if not scored or selection == "union":
        return [box for box, _ in scored]
    return [max(scored, key=lambda item: item[1])[0]]


def encode_frame(
    frame: np.ndarray,
    max_width: int = _LLM_MAX_WIDTH,
    max_height: int = _LLM_MAX_HEIGHT,
) -> str:
    """
    Encode a single frame to base64 in JPG. If the frame is larger than
    `max_width` or `max_height`, it will be resized (aspect-preserving,
    downscale only).
    """
    h, w = frame.shape[:2]
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        frame = cv2.resize(
            frame, 
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )
    _, buf = cv2.imencode(
        ".jpg", frame, 
        [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
    )
    return base64.b64encode(buf).decode()

def compile_video(frames: list[np.ndarray], fps: float) -> bytes:
    """
    Compile multiple `frames` into a single MP4 video. Frames will be resized 
    to `VIDEO_SIZE`.
    """
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

def save_clip_and_data(
    base: Path,
    messages: list[dict],
    frames: list[np.ndarray],
    video_fps: float
) -> None:
    """
    Write a clip's user-prompt JSON and compiled video to `{base}.json`
    and `{base}.mp4`.
    """
    user_content = next(
        (m["content"] for m in messages if m["role"] == "user"),
        []
    )

    # Save JSON file
    try:
        p = base.with_suffix(".json")
        p.write_text(json.dumps(user_content, indent=2))
        p.chmod(0o666)
    except OSError:
        logger.exception("Failed to save clip JSON to %s", base)

    # Save video file
    try:
        video_bytes = compile_video(frames, video_fps)
        p = base.with_suffix(".mp4")
        p.write_bytes(video_bytes)
        p.chmod(0o666)
    except Exception:
        logger.exception("Failed to save clip video to %s", base)

def healthcheck_ping(url: str) -> None:
    """
    Heartbeat to Healthcheck or similar, sent every minute.
    """
    while True:
        try:
            urllib.request.urlopen(url, timeout=10)
        except Exception:
            logger.exception("Healthcheck ping failed")
        time.sleep(60)

def footer_timestamp() -> str:
    """Local-time timestamp with one decimal second, for message footers.

    Returns HTML-formatted (italic) text; senders must use parse_mode="HTML".
    """
    now = time.time()
    ts = f"<i>({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}."
    ts += f"{int(now % 1 * 10)})</i>"
    return ts

def format_age(ts: datetime) -> str:
    """
    Format the age of a timestamp (e.g. last frame time) as a human-readable 
    string, either in seconds, minutes, or hours.
    """
    age = (datetime.now().astimezone() - ts).total_seconds()
    if age < 60:
        return f"{age:.0f}s ago"
    elif age < 3600:
        return f"{age / 60:.0f}m ago"
    else:
        return f"{age / 3600:.1f}h ago"
