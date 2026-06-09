import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import requests

from config import TelegramConfig

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org"
_VIDEO_SIZE = (960, 540)
class TelegramClient:
    def __init__(
        self,
        config: TelegramConfig,
        video_fps: float,
        data_dir: Path,
    ):
        self._url = f"{_API_BASE}/bot{config.bot_token}"
        self._chat_ids = config.chat_ids
        self._alert_threshold = config.alert_threshold
        self._alert_cooldown = config.alert_cooldown
        self._escalation_threshold = config.escalation_threshold
        self._live_stream_url = config.live_stream_url
        self._logs_url = config.logs_url
        self._video_fps = video_fps
        self._last_alert_time = 0.0
        self._last_alert_score = 0
        self._chat_ids_lock = threading.Lock()
        self._thresholds_path = data_dir / "thresholds.json"
        self._thresholds: dict[int, int] = self._load_thresholds()
        self._sysalert_disabled_path = data_dir / "sysalert_disabled.json"
        self._sysalert_disabled: set[int] = self._load_sysalert_disabled()
        self._mute_until: dict[int, float] = {}

    def _load_thresholds(self) -> dict[int, int]:
        try:
            with open(self._thresholds_path) as f:
                return {int(k): v for k, v in json.load(f).items()}
        except FileNotFoundError:
            return {}
        except Exception:
            logger.exception("Failed to load thresholds, starting fresh")
            return {}

    def _save_thresholds(self) -> None:
        try:
            self._thresholds_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._thresholds_path, "w") as f:
                json.dump({str(k): v for k, v in self._thresholds.items()}, f)
        except Exception:
            logger.exception("Failed to save thresholds")

    def _load_sysalert_disabled(self) -> set[int]:
        try:
            with open(self._sysalert_disabled_path) as f:
                return set(json.load(f))
        except FileNotFoundError:
            return set()
        except Exception:
            logger.exception("Failed to load sysalert prefs, starting fresh")
            return set()

    def _save_sysalert_disabled(self) -> None:
        try:
            self._sysalert_disabled_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._sysalert_disabled_path, "w") as f:
                json.dump(list(self._sysalert_disabled), f)
        except Exception:
            logger.exception("Failed to save sysalert prefs")

    def set_sysalert(self, chat_id: int, enabled: bool) -> None:
        with self._chat_ids_lock:
            if enabled:
                self._sysalert_disabled.discard(chat_id)
            else:
                self._sysalert_disabled.add(chat_id)
            self._save_sysalert_disabled()
        logger.info("System alerts for chat %d set to %s", chat_id, enabled)

    def get_sysalert(self, chat_id: int) -> bool:
        with self._chat_ids_lock:
            return chat_id not in self._sysalert_disabled

    def set_threshold(self, chat_id: int, threshold: int) -> None:
        with self._chat_ids_lock:
            self._thresholds[chat_id] = threshold
            self._save_thresholds()
        logger.info("Alert threshold for chat %d set to %d", chat_id, threshold)

    def get_threshold(self, chat_id: int) -> int:
        with self._chat_ids_lock:
            return self._thresholds.get(chat_id, self._alert_threshold)

    def mute(self, chat_id: int, seconds: float) -> None:
        with self._chat_ids_lock:
            self._mute_until[chat_id] = time.monotonic() + seconds
        logger.info("Alerts muted for chat %d for %.0fs", chat_id, seconds)

    def unmute(self, chat_id: int) -> None:
        with self._chat_ids_lock:
            self._mute_until.pop(chat_id, None)
        logger.info("Alerts unmuted for chat %d", chat_id)

    def mute_remaining(self, chat_id: int) -> float:
        with self._chat_ids_lock:
            return max(0.0, self._mute_until.get(chat_id, 0.0) - time.monotonic())

    def update_chat_ids(self, chat_ids: list[int]) -> None:
        with self._chat_ids_lock:
            self._chat_ids = chat_ids
            logger.info("Telegram chat IDs updated")

    def send_alert(self, score: int, summary: str, description: str, frames: list[np.ndarray]) -> None:
        with self._chat_ids_lock:
            now = time.monotonic()
            chat_ids = [cid for cid in self._chat_ids if self._mute_until.get(cid, 0.0) <= now]
            thresholds = {cid: self._thresholds.get(cid, self._alert_threshold) for cid in chat_ids}
        if not chat_ids or score < min(thresholds.values()):
            return
        now = time.monotonic()
        cooldown_expired = (now - self._last_alert_time) >= self._alert_cooldown
        escalated = score >= self._last_alert_score + self._escalation_threshold
        if not cooldown_expired and not escalated:
            return
        self._last_alert_time = now
        self._last_alert_score = score
        text = f"{score} - {summary}\n\n{description}\n\nLive stream: {self._live_stream_url}\nLogs: {self._logs_url}"
        video_bytes = _compile_video(frames, self._video_fps)
        for chat_id in chat_ids:
            if score < thresholds[chat_id]:
                continue
            requests.post(
                f"{self._url}/sendVideo",
                data={"chat_id": chat_id, "caption": text},
                files={"video": ("alert.mp4", video_bytes, "video/mp4")},
                timeout=60,
            ).raise_for_status()

    def send_system_alert(self, description: str) -> None:
        with self._chat_ids_lock:
            now = time.monotonic()
            chat_ids = [
                cid for cid in self._chat_ids
                if cid not in self._sysalert_disabled and self._mute_until.get(cid, 0.0) <= now
            ]
        for chat_id in chat_ids:
            requests.post(
                f"{self._url}/sendMessage",
                data={"chat_id": chat_id, "text": description},
                timeout=60,
            ).raise_for_status()

    def start_polling(self, commands: dict[str, Callable[[int, str], str | tuple[str, list]]]) -> None:
        threading.Thread(target=self._poll_loop, args=(commands,), daemon=True, name="telegram-poll").start()

    def _poll_loop(self, commands: dict[str, Callable[[int, str], str | tuple[str, list]]]) -> None:
        # Discard any pending updates accumulated while offline
        try:
            resp = requests.post(f"{self._url}/getUpdates", json={"offset": -1}, timeout=10)
            updates = resp.json().get("result", [])
            offset = updates[-1]["update_id"] + 1 if updates else 0
        except Exception:
            offset = 0

        while True:
            try:
                resp = requests.post(
                    f"{self._url}/getUpdates",
                    json={"offset": offset, "timeout": 30, "allowed_updates": ["message"]},
                    timeout=35,
                )
                for update in resp.json().get("result", []):
                    offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    chat_id = msg.get("chat", {}).get("id")
                    text = msg.get("text", "")
                    command = text.split()[0] if text.startswith("/") else ""
                    if chat_id and command in commands:
                        try:
                            result = commands[command](chat_id, text)
                            if isinstance(result, tuple):
                                caption, frames = result
                                requests.post(
                                    f"{self._url}/sendMessage",
                                    json={"chat_id": chat_id, "text": caption},
                                    timeout=10,
                                ).raise_for_status()
                                video_bytes = _compile_video(frames, self._video_fps)
                                requests.post(
                                    f"{self._url}/sendVideo",
                                    data={"chat_id": chat_id},
                                    files={"video": ("last.mp4", video_bytes, "video/mp4")},
                                    timeout=60,
                                ).raise_for_status()
                            else:
                                requests.post(
                                    f"{self._url}/sendMessage",
                                    json={"chat_id": chat_id, "text": result},
                                    timeout=10,
                                )
                        except Exception:
                            logger.exception("Failed to handle %s", command)
            except Exception:
                logger.exception("Poll error")
                time.sleep(5)


def _compile_video(frames: list[np.ndarray], fps: float) -> bytes:
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, frame in enumerate(frames):
            resized = cv2.resize(frame, _VIDEO_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(tmp_dir, f"f{i:04d}.jpg"), resized)

        out = os.path.join(tmp_dir, "out.mp4")
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
        with open(out, "rb") as f:
            return f.read()