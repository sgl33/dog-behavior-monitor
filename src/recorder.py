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

logger = logging.getLogger(__name__)


class Recorder(threading.Thread):
    """
    Saves a buffer of recent video frames in memory.
    """

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
        self._telegram_client.register_camera(camera)
        self._fps = config.fps
        self._offline_alert_seconds = config.offline_alert_seconds
        self._stale_stream_seconds = config.stale_stream_seconds
        self._recovery_seconds = config.recovery_seconds
        self._buffer: deque[tuple[datetime, np.ndarray]] = deque(maxlen=config.fps * config.buffer_seconds)
        self._lock = threading.Lock()
        self._latest_boxes: list[tuple[int, int, int, int]] = []
        self._stop_event = threading.Event()

    def run(self) -> None:
        self.run_loop()

    def run_loop(self) -> None:
        # Single source of truth: when did we last decode a real frame. Both the
        # offline alert and the recovery alert are driven purely off the elapsed
        # time since this, so a stream that stays *connectable* but stops
        # delivering frames is still detected as offline. Only this thread
        # touches the connection state, so plain attributes need no locking.
        self._last_good_frame_mono = time.monotonic()
        self._offline_alerted = False
        self._healthy_since: float | None = None

        while not self._stop_event.is_set():
            self._run_once()

    def _run_once(self) -> None:
        """One connect → capture → disconnect cycle, then alert/back-off."""
        cap = None
        try:
            cap = self._open_capture()
            got_frame = self._capture_session(cap) if cap.isOpened() else False
            cap.release()
            cap = None

            self._maybe_alert_offline()
            if not got_frame:
                self._stop_event.wait(5.0)
        except Exception:
            # Never let an unexpected error kill the thread
            logger.exception("%s recorder loop error, retrying", self.camera)
            if cap is not None:
                cap.release()
            self._stop_event.wait(5.0)

    def _open_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self._rtsp_url, cv2.CAP_FFMPEG, [
            # On-demand relays (go2rtc/Frigate restream) can take several
            # seconds to spin up the upstream pull and deliver a keyframe,
            # so allow a generous open timeout. The read timeout stays
            # short — it's what detects a mid-stream stall.
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000,
        ])

    def _capture_session(self, cap: cv2.VideoCapture) -> bool:
        """Pull frames until the stream stalls or we're asked to stop.

        Returns whether any frame was decoded this session.
        """
        got_frame = False
        next_capture = time.monotonic()
        # Session-local stall timer, reset fresh on each new connection. This
        # must NOT use _last_good_frame_mono: after an outage that one is
        # arbitrarily old, so it would trip on the first iteration and break
        # before we ever read a frame — permanently preventing reconnection.
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
            prev_good_mono = self._last_good_frame_mono
            self._last_good_frame_mono = now
            session_last_frame = now
            self._store_frame(frame)
            next_capture = now + (1.0 / self._fps)
            if self._offline_alerted:
                self._update_recovery(now, prev_good_mono)
        return got_frame

    def _store_frame(self, frame: np.ndarray) -> None:
        # Copy outside the lock — the full-frame memcpy is non-trivial and frame
        # is thread-local, so only the append needs guarding. JPEG encoding is
        # deferred to sampling time (see _build_frame_content): every frame here
        # is buffered but only the handful actually sent to the LLM get encoded.
        item = (datetime.now(), frame.copy())
        with self._lock:
            self._buffer.append(item)

    def _update_recovery(self, now: float, prev_good_mono: float) -> None:
        """While offline, watch for a sustained run of frames before declaring
        recovery, so a flapping stream that trickles the odd frame doesn't flip
        online/offline repeatedly."""
        # Any gap longer than the stale threshold restarts the clock.
        if self._healthy_since is None or now - prev_good_mono > self._stale_stream_seconds:
            self._healthy_since = now
            return
        if now - self._healthy_since < self._recovery_seconds:
            return
        logger.info("%s camera back online", self.camera)
        # Commit local state before the network send so a failed alert can't
        # leave _offline_alerted stuck True (which would re-alert forever).
        self._offline_alerted = False
        self._healthy_since = None
        self._telegram_client.set_camera_online(self.camera, True)
        self._telegram_client.send_system_alert(
            f"✅ [{self.camera}] camera back online - {self._telegram_client.camera_status_summary()}",
            silent=True,
        )

    def _maybe_alert_offline(self) -> None:
        if self._offline_alerted:
            return
        if time.monotonic() - self._last_good_frame_mono < self._offline_alert_seconds:
            return
        logger.warning("%s camera offline", self.camera)
        # Commit local state before the network send: if the alert throws,
        # _offline_alerted must still flip True so this camera stops re-alerting
        # and can later be recovered to online.
        self._offline_alerted = True
        self._healthy_since = None
        self._telegram_client.set_camera_online(self.camera, False)
        self._telegram_client.send_system_alert(
            f"📵 [{self.camera}] camera offline - {self._telegram_client.camera_status_summary()}",
            silent=True,
        )

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

    def get_frames(
        self, last_seconds: float
    ) -> list[tuple[datetime, np.ndarray]]:
        """
        Get most recent (timestamp, frame data) frames.
        """
        cutoff = datetime.now()
        with self._lock:
            return [
                item for item in self._buffer
                if (cutoff - item[0]).total_seconds() <= last_seconds
            ]

    def stop(self) -> None:
        self._stop_event.set()
