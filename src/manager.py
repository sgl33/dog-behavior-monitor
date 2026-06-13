import json
import logging
import threading
import time
from datetime import datetime

import numpy as np

from config import Config
from eval_saver import EvalSaver
from llm import LLMClient, extract_json
from llm_logger import LLMOutputLogger
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
        config: Config,
        llm_logger: LLMOutputLogger | None = None,
        eval_saver: EvalSaver | None = None,
    ):
        super().__init__(daemon=True, name="manager")
        self._cameras = cameras
        self._state = state
        self._recorders = recorders
        self._llm_client = llm_client
        self._telegram_client = telegram_client
        self._web_server = web_server
        self._detection_window = config.llm_endpoint.detection_window
        self._post_llm_cooldown = config.post_llm_cooldown
        self._slow_threshold = config.llm_endpoint.slow_threshold
        self._alert_threshold = config.telegram.alert_threshold
        self._no_detection_interval = config.no_detection_fallback_seconds
        self._fallback_detection_enabled = config.fallback_detection_enabled
        self._llm_enabled = True
        self._llm_logger = llm_logger
        self._eval_saver = eval_saver
        self._llm_busy = threading.Event()
        self._last_llm_time = 0.0
        self._last_llm_finish_time = 0.0
        self._last_llm_inference_latency: float | None = None
        self._llm_slow = False
        self._llm_error = False
        self._llm_consecutive_errors = 0
        self._last_result: tuple[int, str, datetime] | None = None
        self._last_frames: list[np.ndarray] | None = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            now = time.monotonic()
            if not self._llm_enabled:
                self._stop_event.wait(0.5)
                continue
            if self._state.any_recent(within_seconds=self._detection_window):
                if not self._llm_busy.is_set() and (now - self._last_llm_finish_time) >= self._post_llm_cooldown:
                    logger.info("Dog detected, firing LLM")
                    self._fire_llm(fallback=False)
            elif self._fallback_detection_enabled and not self._llm_busy.is_set() and (now - self._last_llm_time) >= self._no_detection_interval:
                logger.info("No dog detected, firing fallback LLM")
                self._fire_llm(fallback=True)

            self._stop_event.wait(0.05)

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
            if self._llm_error:
                self._llm_error = False
                self._llm_consecutive_errors = 0
                self._telegram_client.send_system_alert("✅ LLM recovered")
        except Exception as e:
            logger.exception("Fallback check error")
            self._llm_consecutive_errors += 1
            if self._llm_consecutive_errors >= 3 and not self._llm_error:
                self._llm_error = True
                self._telegram_client.send_system_alert(f"⚠️ LLM error: {e}")
            self._llm_busy.clear()

    def _run_llm(self, frames_by_camera: dict[str, list[tuple[datetime, np.ndarray]]], boxes_by_camera: dict[str, list[tuple[int, int, int, int]]], fire_time: float, detected_by: str = "YOLO") -> None:
        try:
            if not any(frames_by_camera.values()):
                logger.info("No frames available, skipping LLM inference")
                return
            logger.info("LLM inference started")
            response, frames, messages = self._llm_client.analyze(frames_by_camera, boxes_by_camera)
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
            logger.info("LLM result (pass 1): %d - %s (%.2fs)", score, description, self._last_llm_inference_latency)

            double_pass = False
            if score >= self._alert_threshold:
                logger.info("Score %d >= threshold %d, running second pass to verify", score, self._alert_threshold)
                response2, frames2, messages2 = self._llm_client.analyze(frames_by_camera, boxes_by_camera)
                parsed2 = json.loads(extract_json(response2))
                score, summary, description = parsed2["score"], parsed2["summary"], parsed2["description"]
                frames, messages = frames2, messages2
                double_pass = True
                logger.info("LLM result (pass 2): %d - %s", score, description)

            result_time = datetime.now().astimezone()
            self._last_result = (score, description, result_time)
            self._last_frames = frames
            if self._web_server is not None:
                self._web_server.push_result(score, summary, description, result_time, frames, self._last_llm_inference_latency, list(frames_by_camera.keys()), detected_by, double_pass)
            logger.info("LLM final result: %d - %s (%.2fs)", score, description, self._last_llm_inference_latency)
            if self._llm_logger is not None:
                self._llm_logger.log(result_time, score, summary, description, self._last_llm_inference_latency, list(frames_by_camera.keys()), detected_by)
            if self._llm_error:
                self._llm_error = False
                self._llm_consecutive_errors = 0
                self._telegram_client.send_system_alert("✅ LLM recovered")
            if self._eval_saver is not None:
                self._eval_saver.maybe_save(score, messages, frames)
            self._telegram_client.send_alert(score, summary, description, frames, messages)
        except Exception as e:
            logger.exception("LLM error")
            self._llm_consecutive_errors += 1
            if self._llm_consecutive_errors >= 3 and not self._llm_error:
                self._llm_error = True
                detail = ""
                status = ""
                resp = getattr(e, "response", None)
                if resp is not None:
                    status = str(resp.status_code)
                    try:
                        detail = resp.json().get("error", {}).get("message", "") or ""
                    except Exception:
                        pass
                if detail and status:
                    msg = f"{detail} (HTTP code {status})"
                elif detail:
                    msg = detail
                elif status:
                    msg = f"{status} {getattr(resp, 'reason', '')}".strip()
                else:
                    msg = str(e)
                self._telegram_client.send_system_alert(f"⚠️ LLM error: {msg}")
        finally:
            self._last_llm_finish_time = time.monotonic()
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

    def set_llm_enabled(self, enabled: bool) -> None:
        self._llm_enabled = enabled

    @property
    def llm_enabled(self) -> bool:
        return self._llm_enabled

    def set_fallback_detection_enabled(self, enabled: bool) -> None:
        self._fallback_detection_enabled = enabled

    def set_post_llm_cooldown(self, cooldown: float) -> None:
        self._post_llm_cooldown = cooldown

    def set_detection_window(self, seconds: float) -> None:
        self._detection_window = seconds

    def set_slow_threshold(self, seconds: float) -> None:
        self._slow_threshold = seconds

    def set_no_detection_interval(self, seconds: float) -> None:
        self._no_detection_interval = seconds

    def stop(self) -> None:
        self._stop_event.set()


