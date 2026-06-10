import logging
import os
import threading
import time
from collections import deque
from datetime import datetime

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np

from config import RecorderConfig
from telegram import TelegramClient
from utils import encode_frame

logger = logging.getLogger(__name__)


class Recorder(threading.Thread):
    def __init__(
        self,
        camera: str,
        rtsp_url: str,
        telegram_client: TelegramClient,
        config: RecorderConfig,
    ):
        super().__init__(daemon=True, name=f"recorder-{camera}")
        self.camera = camera
        self._rtsp_url = rtsp_url
        self._telegram_client = telegram_client
        self._fps = config.fps
        self._offline_alert_seconds = config.offline_alert_seconds
        self._stale_stream_seconds = config.stale_stream_seconds
        self._buffer: deque[tuple[datetime, np.ndarray, str]] = deque(maxlen=config.fps * config.buffer_seconds)
        self._lock = threading.Lock()
        self._latest_boxes: list[tuple[int, int, int, int]] = []
        self._stop_event = threading.Event()

    def run(self) -> None:
        offline_since: float | None = None
        offline_alerted = False

        while not self._stop_event.is_set():
            cap = cv2.VideoCapture(self._rtsp_url, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5_000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000,
            ])
            if not cap.isOpened():
                if offline_since is None:
                    offline_since = time.monotonic()
                if not offline_alerted and time.monotonic() - offline_since >= self._offline_alert_seconds:
                    logger.warning("%s camera offline", self.camera)
                    self._telegram_client.send_system_alert(f"📵 [{self.camera}] camera offline")
                    offline_alerted = True
                self._stop_event.wait(5.0)
                continue

            received_frame = False
            next_capture = time.monotonic()
            last_frame_mono = time.monotonic()
            while not self._stop_event.is_set():
                now = time.monotonic()
                if now - last_frame_mono > self._stale_stream_seconds:
                    if offline_since is None:
                        offline_since = time.monotonic()
                    break
                if now < next_capture:
                    if not cap.grab():
                        if offline_since is None:
                            offline_since = time.monotonic()
                        break
                    continue
                ret, frame = cap.read()
                if not ret:
                    if offline_since is None:
                        offline_since = time.monotonic()
                    break
                if offline_alerted:
                    logger.info("%s camera back online", self.camera)
                    self._telegram_client.send_system_alert(f"✅ [{self.camera}] camera back online")
                    offline_alerted = False
                offline_since = None
                received_frame = True
                last_frame_mono = now
                with self._lock:
                    f = frame.copy()
                    self._buffer.append((datetime.now(), f, encode_frame(f)))
                next_capture = now + (1.0 / self._fps)

            cap.release()
            if not received_frame:
                if not offline_alerted and offline_since is not None and time.monotonic() - offline_since >= self._offline_alert_seconds:
                    logger.warning("%s camera offline", self.camera)
                    self._telegram_client.send_system_alert(f"📵 [{self.camera}] Camera offline")
                    offline_alerted = True
                self._stop_event.wait(5.0)

    def set_latest_boxes(self, boxes: list[tuple[int, int, int, int]]) -> None:
        with self._lock:
            self._latest_boxes = boxes

    @property
    def latest_boxes(self) -> list[tuple[int, int, int, int]]:
        with self._lock:
            return list(self._latest_boxes)

    def last_frame_time(self) -> datetime | None:
        with self._lock:
            return self._buffer[-1][0] if self._buffer else None

    def get_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._buffer[-1][1] if self._buffer else None

    def get_frames(self, last_seconds: float) -> list[tuple[datetime, np.ndarray, str]]:
        cutoff = datetime.now()
        with self._lock:
            return [
                item for item in self._buffer
                if (cutoff - item[0]).total_seconds() <= last_seconds
            ]

    def stop(self) -> None:
        self._stop_event.set()
