import json
import logging
import threading
import time
from datetime import datetime

import numpy as np

from llm import LLMClient, extract_json
from recorder import Recorder
from state import DogDetectionState
from telegram import TelegramClient
from web_server import WebServerClient

logger = logging.getLogger(__name__)


class Manager(threading.Thread):
    def __init__(
        self,
        cameras: list[str],
        state: DogDetectionState,
        recorders: dict[str, Recorder],
        llm_client: LLMClient,
        telegram_client: TelegramClient,
        web_server: WebServerClient | None,
        detection_window: float,
        llm_cooldown: float,
        loop_interval: float,
        slow_threshold: float,
        no_detection_interval: float,
    ):
        super().__init__(daemon=True, name="manager")
        self._cameras = cameras
        self._state = state
        self._recorders = recorders
        self._llm_client = llm_client
        self._telegram_client = telegram_client
        self._web_server = web_server
        self._detection_window = detection_window
        self._llm_cooldown = llm_cooldown
        self._loop_interval = loop_interval
        self._slow_threshold = slow_threshold
        self._no_detection_interval = no_detection_interval
        self._llm_busy = threading.Event()
        self._last_llm_time = 0.0
        self._last_llm_inference_latency: float | None = None
        self._llm_slow = False
        self._llm_error = False
        self._fallback_error = False
        self._last_result: tuple[int, str, datetime] | None = None
        self._last_frames: list[np.ndarray] | None = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            now = time.monotonic()
            if self._state.any_recent(within_seconds=self._detection_window):
                if not self._llm_busy.is_set() and (now - self._last_llm_time) >= self._llm_cooldown:
                    logger.info("Dog detected, firing LLM")
                    self._fire_llm(fallback=False)
            elif not self._llm_busy.is_set() and (now - self._last_llm_time) >= self._no_detection_interval:
                logger.info("No dog detected, firing fallback LLM")
                self._fire_llm(fallback=True)
            else:
                logger.debug("Dog not detected")

            self._stop_event.wait(self._loop_interval)

    def _fire_llm(self, fallback: bool = False) -> None:
        self._llm_busy.set()
        self._last_llm_time = time.monotonic()
        if fallback:
            latest_frames = {
                cam: frame
                for cam in self._cameras
                if (frame := self._recorders[cam].get_latest_frame()) is not None
            }
            threading.Thread(target=self._run_fallback_check, args=(latest_frames,), daemon=True).start()
        else:
            cameras = self._state.recent_cameras(within_seconds=self._detection_window)
            frames_by_camera = {
                cam: self._recorders[cam].get_frames(last_seconds=self._detection_window)
                for cam in cameras
            }
            boxes_by_camera = {cam: self._recorders[cam].latest_boxes for cam in cameras}
            fire_time = time.monotonic()
            threading.Thread(target=self._run_llm, args=(frames_by_camera, boxes_by_camera, fire_time, "YOLO"), daemon=True).start()

    def _run_fallback_check(self, latest_frames: dict[str, np.ndarray]) -> None:
        try:
            cameras_with_dog = self._llm_client.detect_dog(latest_frames)
            if cameras_with_dog:
                if self._state.any_recent(within_seconds=self._detection_window):
                    logger.info("Fallback detected dog in %s, YOLO confirmed — main loop will handle", cameras_with_dog)
                    self._llm_busy.clear()
                else:
                    logger.info("Fallback detected dog in %s, no YOLO confirmation — firing full LLM", cameras_with_dog)
                    frames_by_camera = {
                        cam: self._recorders[cam].get_frames(last_seconds=self._detection_window)
                        for cam in cameras_with_dog
                    }
                    self._run_llm(frames_by_camera, {}, time.monotonic(), "LLM")
            else:
                logger.debug("Fallback: no dog found")
                self._llm_busy.clear()
            if self._fallback_error:
                self._fallback_error = False
                self._telegram_client.send_system_alert("✅ Fallback LLM recovered")
        except Exception as e:
            logger.exception("Fallback check error")
            if not self._fallback_error:
                self._fallback_error = True
                self._telegram_client.send_system_alert(f"⚠️ Fallback check error: {e}")
            self._llm_busy.clear()

    def _run_llm(self, frames_by_camera: dict[str, list[tuple[datetime, np.ndarray]]], boxes_by_camera: dict[str, list[tuple[int, int, int, int]]], fire_time: float, detected_by: str = "YOLO") -> None:
        try:
            logger.info("LLM inference started")
            response, frames = self._llm_client.analyze(frames_by_camera, boxes_by_camera)
            self._last_llm_inference_latency = time.monotonic() - fire_time
            if self._last_llm_inference_latency > self._slow_threshold:
                if not self._llm_slow:
                    self._llm_slow = True
                    self._telegram_client.send_system_alert(f"⚠️ LLM inference slow: {self._last_llm_inference_latency:.1f}s")
            elif self._llm_slow:
                self._llm_slow = False
                self._telegram_client.send_system_alert(f"✅ LLM inference back to normal: {self._last_llm_inference_latency:.1f}s")

            parsed = json.loads(extract_json(response))
            score, summary, description = parsed["score"], parsed["summary"], parsed["description"]
            result_time = datetime.now()
            self._last_result = (score, description, result_time)
            self._last_frames = frames
            if self._web_server is not None:
                self._web_server.push_result(score, summary, description, result_time, frames, self._last_llm_inference_latency, list(frames_by_camera.keys()), detected_by)
            logger.info("LLM result: %d - %s (%.2fs)", score, description, self._last_llm_inference_latency)
            if self._llm_error:
                self._llm_error = False
                self._telegram_client.send_system_alert("✅ LLM recovered")
            self._telegram_client.send_alert(score, summary, description, frames)
        except Exception as e:
            logger.exception("LLM error")
            if not self._llm_error:
                self._llm_error = True
                self._telegram_client.send_system_alert(f"⚠️ LLM error: {e}")
        finally:
            self._llm_busy.clear()

    @property
    def last_llm_inference_latency(self) -> float | None:
        return self._last_llm_inference_latency

    @property
    def last_result(self) -> tuple[int, str, datetime] | None:
        return self._last_result

    @property
    def last_frames(self) -> list[np.ndarray] | None:
        return self._last_frames

    def trigger_now(self) -> str:
        if self._llm_busy.is_set():
            return "⏳ LLM is already running, please wait"
        cameras_with_frames = [
            cam for cam in self._cameras
            if self._recorders[cam].last_frame_time() is not None
        ]
        if not cameras_with_frames:
            return "❌ No frames available yet"
        self._llm_busy.set()
        self._last_llm_time = time.monotonic()
        frames_by_camera = {
            cam: self._recorders[cam].get_frames(last_seconds=self._detection_window)
            for cam in cameras_with_frames
        }
        boxes_by_camera = {cam: self._recorders[cam].latest_boxes for cam in cameras_with_frames}
        fire_time = time.monotonic()
        threading.Thread(target=self._run_llm, args=(frames_by_camera, boxes_by_camera, fire_time, "YOLO"), daemon=True).start()
        return "✅ LLM analysis triggered"

    def stop(self) -> None:
        self._stop_event.set()


