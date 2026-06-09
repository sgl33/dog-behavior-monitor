import logging
import threading
import time

import numpy as np
from ultralytics import YOLO

from config import Config
from recorder import Recorder
from state import DogDetectionState
from telegram import TelegramClient

logger = logging.getLogger(__name__)

_DOG_CLASS_ID = 16


class Detector(threading.Thread):
    def __init__(
        self,
        camera: str,
        recorder: Recorder,
        state: DogDetectionState,
        model: YOLO,
        model_lock: threading.Lock,
        telegram_client: TelegramClient,
        config: Config,
    ):
        super().__init__(daemon=True, name=f"detector-{camera}")
        self.camera = camera
        self._recorder = recorder
        self._state = state
        self._detect_interval = config.detect_interval
        self._model = model
        self._model_lock = model_lock
        self._device = config.yolo_device
        self._image_size = config.yolo_image_size
        self._telegram_client = telegram_client
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            start = time.monotonic()

            frame = self._recorder.get_latest_frame()
            if frame is not None:
                inference_start = time.monotonic()
                self._run_inference(frame)
                inference_end = time.monotonic()
                if (inference_end - inference_start > self._detect_interval):
                    logger.warning("[%s] YOLO inference falling behind: %.1fs (interval: %.1fs)", self.camera, inference_end - inference_start, self._detect_interval)
                    self._telegram_client.send_system_alert(f"⚠️ [{self.camera}] YOLO inference falling behind: {inference_end - inference_start:.1f}s (interval: {self._detect_interval}s)")

            elapsed = time.monotonic() - start
            self._stop_event.wait(max(0.0, self._detect_interval - elapsed))

    def _run_inference(self, frame: np.ndarray) -> None:
        with self._model_lock:
            results = self._model.predict(frame, device=self._device, imgsz=self._image_size, verbose=False)
        boxes = results[0].boxes
        dog_boxes = [
            (int(x1), int(y1), int(x2), int(y2))
            for (x1, y1, x2, y2), cls in zip(boxes.xyxy.tolist(), boxes.cls.tolist())
            if int(cls) == _DOG_CLASS_ID
        ]
        self._recorder.set_latest_boxes(dog_boxes)
        if dog_boxes:
            self._state.update(self.camera)

    def stop(self) -> None:
        self._stop_event.set()
