import logging
import threading
import time
from datetime import datetime

import numpy as np
from ultralytics import YOLO

from config import Config
from recorder import Recorder
from state import DogDetectionState
from telegram import TelegramClient

logger = logging.getLogger(__name__)

_DOG_CLASS_ID = 16


class YoloLagMonitor:
    """
    Tracks YOLO inference lag alert state shared across all detectors.

    Alerts if the most recent YOLO inference took longer than the configured
    detection interval, and recovers when it goes back within the interval.
    """

    # Only alert on "falling behind" once it has been sustained this long, to
    # avoid noise from brief one-off spikes.
    _BEHIND_GRACE_SECONDS = 2.0

    def __init__(self, detect_interval: float, telegram_client: TelegramClient):
        self._detect_interval = detect_interval
        self._telegram_client = telegram_client
        self._lock = threading.Lock()
        self._alerted_behind = False
        self._alerted_critical = False
        self._behind_since: float | None = None
        self._last_elapsed: float | None = None
        self._last_time: datetime | None = None

    def record(self, camera: str, elapsed: float) -> None:
        """
        Report one inference's elapsed time and send a Telegram alert if the
        shared lag state changed.
        """
        with self._lock:
            self._last_elapsed = elapsed
            self._last_time = datetime.now()

            now_mono = time.monotonic()

            # Exceeds 3x interval
            if elapsed > self._detect_interval * 3:  # exceeds 3x interval
                if not self._alerted_critical:
                    self._alerted_critical = True
                    self._alerted_behind = True
                    msg = f"🔴 [{camera}] YOLO inference critically behind: {elapsed:.2f}s "
                    msg += f"(3x interval: {self._detect_interval * 3}s)"
                    logger.warning(msg)
                    self._telegram_client.send_system_alert(msg)
            # Exceeds 1x interval
            elif elapsed > self._detect_interval:
                if self._behind_since is None:
                    self._behind_since = now_mono
                # Only alert once it's been falling behind for the grace period
                if not self._alerted_behind and now_mono - self._behind_since > self._BEHIND_GRACE_SECONDS:
                    self._alerted_behind = True
                    msg = f"⚠️ [{camera}] YOLO inference falling behind: {elapsed:.2f}s "
                    msg += f"(interval: {self._detect_interval}s)"
                    logger.warning(msg)
                    self._telegram_client.send_system_alert(msg)
            # Recovered under interval
            else:
                self._behind_since = None
                if self._alerted_behind:
                    self._alerted_behind = False
                    self._alerted_critical = False
                    msg = f"✅ [{camera}] YOLO inference recovered: {elapsed:.2f}s (interval: {self._detect_interval}s)"
                    logger.info(msg)
                    self._telegram_client.send_system_alert(msg, silent=True)

    @property
    def last_inference(self) -> tuple[float, datetime] | None:
        """Most recent YOLO inference duration and when it finished, if any."""
        with self._lock:
            if self._last_elapsed is None or self._last_time is None:
                return None
            return self._last_elapsed, self._last_time


class Detector(threading.Thread):
    """
    Periodically fetches frame from camera, runs it through YOLO for object
    detection, and updates the state to the manager.
    """

    def __init__(
        self,
        camera_name: str,
        recorder: Recorder,
        state: DogDetectionState,
        model: YOLO,
        model_lock: threading.Lock,
        telegram_client: TelegramClient,
        config: Config,
        lag_monitor: YoloLagMonitor,
    ):
        super().__init__(daemon=True, name=f"detector-{camera_name}")
        self.camera = camera_name
        self._recorder = recorder
        self._state = state
        self._detect_interval = config.detect_interval
        self._model = model
        self._model_lock = model_lock
        self._device = config.yolo_device
        self._image_size = config.yolo_image_size
        self._telegram_client = telegram_client
        self._stop_event = threading.Event()
        self._lag_monitor = lag_monitor

    def run(self) -> None:
        """
        Loop that runs forever to grab frames and detect YOLO.
        """
        while not self._stop_event.is_set():
            start = time.monotonic()

            # Fetch frame
            frame = self._recorder.get_latest_frame()
            if frame is not None:
                # Run inference
                inference_start = time.monotonic()
                self._run_inference(frame)
                inference_end = time.monotonic()

                # YOLO inference lag alerts (shared across all detectors)
                self._lag_monitor.record(
                    camera=self.camera,
                    elapsed=(inference_end - inference_start)
                )

            elapsed = time.monotonic() - start
            self._stop_event.wait(max(0.0, self._detect_interval - elapsed))

    def _run_inference(self, frame: np.ndarray) -> None:
        """
        Run YOLO object detection inference (not LLM inference) and submit the
        results to the manager (via `self._state`).
        """
        with self._model_lock:
            results = self._model.predict(
                frame, device=self._device,
                imgsz=self._image_size, half=True, verbose=False
            )
        boxes = results[0].boxes
        dog_boxes = [
            (int(x1), int(y1), int(x2), int(y2))
            for (x1, y1, x2, y2), cls 
            in zip(boxes.xyxy.tolist(), boxes.cls.tolist())
            if int(cls) == _DOG_CLASS_ID
        ]
        self._recorder.set_latest_boxes(dog_boxes)
        if dog_boxes:
            self._state.update(self.camera)

    def set_detect_interval(self, interval: float) -> None:
        self._detect_interval = interval

    def stop(self) -> None:
        self._stop_event.set()
