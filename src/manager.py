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
        self._analysis_window = config.llm_endpoint.analysis_window
        self._post_llm_cooldown = config.llm_endpoint.cooldown
        self._min_interval = config.llm_endpoint.min_interval
        self._slow_threshold = config.llm_endpoint.slow_threshold
        self._alert_threshold = config.telegram.alert_threshold
        self._no_detection_interval = config.no_detection_fallback_seconds
        self._fallback_detection_enabled = config.fallback_detection_enabled
        self._double_pass_enabled = config.double_pass.enabled
        self._llm_enabled = True
        self._llm_logger = llm_logger
        self._eval_saver = eval_saver
        self._max_concurrent = config.llm_endpoint.max_concurrent
        self._llm_lock = threading.Lock()
        self._llm_in_flight = 0
        self._last_llm_time = 0.0
        self._last_llm_finish_time = 0.0
        self._last_llm_inference_latency: float | None = None
        self._last_llm_finish_wall_time: datetime | None = None
        self._llm_slow = False
        self._llm_error = False
        self._llm_consecutive_errors = 0
        self._last_result: tuple[int, str, datetime] | None = None
        self._last_frames: list[np.ndarray] | None = None
        self._stop_event = threading.Event()

    def run(self) -> None:
        """
        Loops forever, checking if any detectors detected a dog recently and
        trigger LLM if applicable.
        """
        while not self._stop_event.is_set():
            now = time.monotonic()
            if not self._llm_enabled:
                self._stop_event.wait(0.5)
                continue

            if self._state.any_recent(within_seconds=self._detection_window):
                if self._llm_in_flight < self._max_concurrent and (now - self._last_llm_finish_time) >= self._post_llm_cooldown and (now - self._last_llm_time) >= self._min_interval:
                    logger.info("Dog detected by YOLO, triggering LLM")
                    self._trigger_llm()
            elif self._fallback_detection_enabled and self._llm_in_flight < self._max_concurrent and (now - self._last_llm_time) >= self._no_detection_interval:
                logger.info("No dog detected, triggering fallback LLM")
                self._trigger_fallback_llm()

            self._stop_event.wait(0.05)

    def _acquire_slot(self) -> None:
        """Reserve an in-flight LLM slot. Callers must gate on capacity first."""
        with self._llm_lock:
            self._llm_in_flight += 1

    def _release_slot(self) -> None:
        """Release an in-flight LLM slot held by a worker thread."""
        with self._llm_lock:
            self._llm_in_flight = max(0, self._llm_in_flight - 1)

    def _llm_worker(self, fn, *args) -> None:
        """
        Run an LLM work function in a worker thread and always release the
        in-flight slot when it returns, regardless of outcome.
        """
        try:
            fn(*args)
        finally:
            self._release_slot()

    def _trigger_llm(self) -> None:
        """
        Run recent video clip(s) through the LLM.
        """
        self._acquire_slot()
        self._last_llm_time = time.monotonic()
        cameras = self._state.recent_cameras(
            within_seconds=self._detection_window
        )
        frames_by_camera = {
            cam: self._recorders[cam].get_frames(
                last_seconds=self._analysis_window
            ) for cam in cameras
        }
        boxes_by_camera = {cam: self._recorders[cam].latest_boxes for cam in cameras}

        fire_time = time.monotonic()
        threading.Thread(
            target=self._llm_worker,
            args=(self._run_llm, frames_by_camera, boxes_by_camera, fire_time, "YOLO"),
            daemon=True
        ).start()

    def _trigger_fallback_llm(self) -> None:
        """
        Trigger fallback object detection when YOLO didn't detect a dog for
        a while.
        """
        latest_frames = {
            cam: frame
            for cam in self._cameras
            if (frame := self._recorders[cam].get_latest_frame()) is not None
        }
        self._last_llm_time = time.monotonic()
        if not latest_frames:
            logger.debug("Fallback: no frames available, skipping")
            return
        self._acquire_slot()
        threading.Thread(
            target=self._llm_worker, args=(self._run_fallback_check, latest_frames),
            daemon=True
        ).start()

    def _run_fallback_check(self, latest_frames: dict[str, np.ndarray]) -> None:
        """
        Use LLM to detect any dogs from images (one per camera), and if any
        are detected, send the video clips through the same LLM.
        """
        try:
            cameras_with_dog = self._llm_client.detect_dog(
                latest_frames,
                should_abort=lambda: self._state.any_recent(within_seconds=self._detection_window),
            )
            if cameras_with_dog:
                if self._state.any_recent(within_seconds=self._detection_window):
                    logger.info("Fallback detected dog in %s, YOLO confirmed — main loop will handle", cameras_with_dog)
                else:
                    logger.info("Fallback detected dog in %s, no YOLO confirmation — firing full LLM", cameras_with_dog)
                    frames_by_camera = {
                        cam: self._recorders[cam].get_frames(last_seconds=self._analysis_window)
                        for cam in cameras_with_dog
                    }
                    self._run_llm(frames_by_camera, {}, time.monotonic(), "LLM")
            else:
                logger.debug("Fallback: no dog found")
            self._handle_llm_recovered()
        except Exception as e:
            logger.exception("Fallback check error")
            self._handle_llm_error(e)

    def _run_llm(
        self, 
        frames_by_camera: dict[str, list[tuple[datetime, np.ndarray]]], 
        boxes_by_camera: dict[str, list[tuple[int, int, int, int]]],
        trigger_time: float,
        detected_by: str = "YOLO"
    ) -> None:
        try:
            if not any(frames_by_camera.values()):
                logger.info("No frames available, skipping LLM inference")
                return
            logger.info("LLM inference started")

            # Run LLM inference
            response, frames, frames_by_camera_sampled, messages, stats = self._llm_client.analyze(frames_by_camera, boxes_by_camera)
            self._last_llm_inference_latency = time.monotonic() - trigger_time
            self._check_inference_slow(self._last_llm_inference_latency)
            parsed = json.loads(extract_json(response))
            score, summary, description = parsed["score"], parsed["summary"], parsed["description"]
            logger.info("LLM result (pass 1): %d - %s (%.2fs)", score, description, self._last_llm_inference_latency)

            # If above global threshold, re-run it to reduce false positives
            double_pass = False
            verify_failed = False
            if self._double_pass_enabled and score >= self._alert_threshold:
                logger.info(
                    "Score %d >= threshold %d, running second pass to verify",
                    score, self._alert_threshold
                )
                # Re-fetch the most recent frames captured while the first pass
                # was running, so the verify pass sees fresh footage (roughly the
                # last `analysis_window` seconds up to now) rather than
                # re-analyzing the identical frames. Boxes are refreshed too so
                # crops track the dog's current position. Fall back to the
                # original frames if the buffer has nothing newer.
                verify_frames_by_camera = {
                    cam: self._recorders[cam].get_frames(last_seconds=self._analysis_window)
                    for cam in frames_by_camera
                    if cam in self._recorders
                }
                verify_boxes_by_camera = {
                    cam: self._recorders[cam].latest_boxes
                    for cam in verify_frames_by_camera
                }
                if not any(verify_frames_by_camera.values()):
                    verify_frames_by_camera = frames_by_camera
                    verify_boxes_by_camera = boxes_by_camera
                # The verify pass only trims false positives, so a failure here
                # must not swallow a pass 1 detection: keep pass 1's result and
                # alert on it, flagged as unverified.
                try:
                    # The displayed inference_time covers pass 1 only, so we keep
                    # pass 1's token/timing stats for the throughput figures and
                    # discard the verify pass's.
                    response2, frames2, frames_by_camera_sampled2, messages2, _ = self._llm_client.analyze(
                        verify_frames_by_camera, verify_boxes_by_camera, verify=True
                    )
                    parsed2 = json.loads(extract_json(response2))
                    score, summary, description = parsed2["score"], parsed2["summary"], parsed2["description"]
                    frames, messages = frames2, messages2
                    frames_by_camera_sampled = frames_by_camera_sampled2
                    double_pass = True
                    logger.info("LLM result (pass 2): %d - %s", score, description)
                except Exception:
                    verify_failed = True
                    logger.exception(
                        "Verify pass failed, alerting on pass 1 result: %d - %s",
                        score, description
                    )

            result_time = datetime.now().astimezone()
            self._last_result = (score, description, result_time)
            self._last_frames = frames
            self._last_llm_finish_time = time.monotonic()
            self._last_llm_finish_wall_time = result_time

            # Push result to web server, logger, and eval clip saver.
            # Send only the most recent frame from each camera for display.
            # Use the cropped sampled frames (same ones compiled into the clips)
            # so the thumbnails match the clip and are cropped to the dog rather
            # than showing the full raw frame.
            if self._web_server is not None:
                web_frames = [
                    cam_frames[-1]
                    for cam_frames in frames_by_camera_sampled.values()
                    if cam_frames
                ]
                self._web_server.push_result(
                    score, summary, description, result_time, web_frames,
                    self._last_llm_inference_latency,
                    list(frames_by_camera.keys()), detected_by, double_pass,
                    clip_frames_by_camera=frames_by_camera_sampled,
                    input_tokens=stats.get("prompt_tokens"),
                    output_tokens=stats.get("completion_tokens"),
                    prefill_time=stats.get("prefill_time"),
                    generation_time=stats.get("generation_time"),
                )
            if self._eval_saver is not None:
                self._eval_saver.save_alert(score, messages, frames)
                self._eval_saver.save_negative(score, messages, frames)
            if self._llm_logger is not None:
                self._llm_logger.log(result_time, score, summary, description, self._last_llm_inference_latency, list(frames_by_camera.keys()), detected_by)

            # Handle sending alert(s)
            self._handle_llm_recovered()
            alert_summary = f"{summary}*" if verify_failed else summary
            self._telegram_client.send_alert(score, alert_summary, description, frames)
        except Exception as e:
            logger.exception("LLM error")
            time.sleep(2)  # backoff before the next inference attempt
            self._handle_llm_error(e)
            self._last_llm_finish_time = time.monotonic()
            self._last_llm_finish_wall_time = datetime.now().astimezone()

    def _check_inference_slow(self, latency: float) -> None:
        """
        Check if LLM inference was slow and send Telegram alerts as applicable.
        """
        if latency > self._slow_threshold:
            if not self._llm_slow:
                self._llm_slow = True
                self._telegram_client.send_system_alert(f"⚠️ LLM inference slow: {latency:.1f}s")
        elif self._llm_slow:
            self._llm_slow = False
            self._telegram_client.send_system_alert(f"✅ LLM inference back to normal: {latency:.1f}s", silent=True)

    def _handle_llm_error(self, e: Exception) -> None:
        """
        Handle LLM error; Telegram alert sent after 3 consecutive failures
        and only once.
        """
        self._llm_consecutive_errors += 1
        if self._llm_consecutive_errors < 3 or self._llm_error:
            return

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

    def _handle_llm_recovered(self) -> None:
        """
        Handle LLM recovering from error, sending a Telegram alert once.
        """
        if self._llm_error:
            self._llm_error = False
            self._llm_consecutive_errors = 0
            self._telegram_client.send_system_alert("✅ LLM recovered", silent=True)

    @property
    def last_llm_inference_latency(self) -> float | None:
        return self._last_llm_inference_latency

    @property
    def last_llm_finish_wall_time(self) -> datetime | None:
        return self._last_llm_finish_wall_time

    @property
    def last_result(self) -> tuple[int, str, datetime] | None:
        return self._last_result

    @property
    def last_frames(self) -> list[np.ndarray] | None:
        return self._last_frames

    @property
    def llm_enabled(self) -> bool:
        return self._llm_enabled

    def set_llm_enabled(self, enabled: bool) -> None:
        self._llm_enabled = enabled

    def set_fallback_detection_enabled(self, enabled: bool) -> None:
        self._fallback_detection_enabled = enabled

    def set_double_pass_enabled(self, enabled: bool) -> None:
        self._double_pass_enabled = enabled

    def set_cooldown(self, cooldown: float) -> None:
        self._post_llm_cooldown = cooldown

    def set_min_interval(self, seconds: float) -> None:
        self._min_interval = seconds

    def set_max_concurrent(self, count: int) -> None:
        self._max_concurrent = count

    def set_detection_window(self, seconds: float) -> None:
        self._detection_window = seconds

    def set_analysis_window(self, seconds: float) -> None:
        self._analysis_window = seconds

    def set_slow_threshold(self, seconds: float) -> None:
        self._slow_threshold = seconds

    def set_no_detection_interval(self, seconds: float) -> None:
        self._no_detection_interval = seconds

    def stop(self) -> None:
        self._stop_event.set()


