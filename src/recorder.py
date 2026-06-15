import logging
import os
import threading
import time
from collections import deque
from datetime import datetime

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
# analyzeduration (µs) / probesize (bytes): give FFmpeg up to ~10s and ~10MB to
# find H.264 codec parameters. Some cameras connect mid-GOP without sending
# SPS/PPS right away ("Could not find codec parameters ... unspecified size");
# a larger probe window lets the stream open instead of failing fast.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|analyzeduration;10000000|probesize;10000000",
)

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
        self._recovery_seconds = config.recovery_seconds
        self._buffer: deque[tuple[datetime, np.ndarray, str]] = deque(maxlen=config.fps * config.buffer_seconds)
        self._lock = threading.Lock()
        self._latest_boxes: list[tuple[int, int, int, int]] = []
        self._stop_event = threading.Event()

    def run(self) -> None:
        # Single source of truth: when did we last decode a real frame. Both the
        # offline alert and the recovery alert are driven purely off the elapsed
        # time since this, so a stream that stays *connectable* but stops
        # delivering frames is still detected as offline.
        last_good_frame_mono = time.monotonic()
        offline_alerted = False
        healthy_since: float | None = None

        while not self._stop_event.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(self._rtsp_url, cv2.CAP_FFMPEG, [
                    # On-demand relays (go2rtc/Frigate restream) can take several
                    # seconds to spin up the upstream pull and deliver a keyframe,
                    # so allow a generous open timeout. The read timeout stays
                    # short — it's what detects a mid-stream stall.
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15_000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000,
                ])
                got_frame = False
                if cap.isOpened():
                    next_capture = time.monotonic()
                    # Session-local stall timer, reset fresh on each new
                    # connection. The inner loop must NOT use last_good_frame_mono
                    # here: after an outage that one is arbitrarily old, so it
                    # would trip on the first iteration and break before we ever
                    # read a frame — permanently preventing reconnection.
                    session_last_frame = time.monotonic()
                    while not self._stop_event.is_set():
                        now = time.monotonic()
                        if now - session_last_frame > self._stale_stream_seconds:
                            break
                        if now < next_capture:
                            if not cap.grab():
                                break
                            continue
                        ret, frame = cap.read()
                        if not ret:
                            break
                        got_frame = True
                        prev_good_mono = last_good_frame_mono
                        last_good_frame_mono = now
                        session_last_frame = now
                        with self._lock:
                            f = frame.copy()
                            self._buffer.append((datetime.now(), f, encode_frame(f)))
                        next_capture = now + (1.0 / self._fps)
                        if offline_alerted:
                            # Require a sustained run of frames before declaring
                            # recovery, so a flapping stream that trickles the odd
                            # frame doesn't flip online/offline repeatedly. Any gap
                            # longer than the stale threshold restarts the clock.
                            if healthy_since is None or now - prev_good_mono > self._stale_stream_seconds:
                                healthy_since = now
                            elif now - healthy_since >= self._recovery_seconds:
                                logger.info("%s camera back online", self.camera)
                                self._telegram_client.send_system_alert(f"✅ [{self.camera}] camera back online")
                                offline_alerted = False
                                healthy_since = None

                cap.release()
                cap = None

                if not offline_alerted and time.monotonic() - last_good_frame_mono >= self._offline_alert_seconds:
                    logger.warning("%s camera offline", self.camera)
                    self._telegram_client.send_system_alert(f"📵 [{self.camera}] camera offline")
                    offline_alerted = True
                    healthy_since = None

                if not got_frame:
                    self._stop_event.wait(5.0)
            except Exception:
                # Never let an unexpected error kill the thread — otherwise this
                # camera would stop recovering (and stop alerting) until the
                # whole process restarts. Log, back off, and retry.
                logger.exception("%s recorder loop error, retrying", self.camera)
                if cap is not None:
                    cap.release()
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
