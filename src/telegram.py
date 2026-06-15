import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import requests

from config import TelegramConfig
from utils import compile_video

logger = logging.getLogger(__name__)

_API_BASE = "https://api.telegram.org"
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
        self._muted_path = data_dir / "muted.json"
        self._muted: set[int] = self._load_muted()
        self._snooze_until: dict[int, float] = {}
        self._save_alerts = config.save_alerts
        self._alerts_dir = data_dir / "alerts"
        self._alerts_dir.mkdir(exist_ok=True)
        self._alerts_dir.chmod(0o777)

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

    def _load_muted(self) -> set[int]:
        try:
            with open(self._muted_path) as f:
                return set(json.load(f))
        except FileNotFoundError:
            return set()
        except Exception:
            logger.exception("Failed to load muted chats, starting fresh")
            return set()

    def _save_muted(self) -> None:
        try:
            self._muted_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._muted_path, "w") as f:
                json.dump(list(self._muted), f)
        except Exception:
            logger.exception("Failed to save muted chats")

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

    def mute(self, chat_id: int) -> None:
        with self._chat_ids_lock:
            self._muted.add(chat_id)
            self._save_muted()
        logger.info("Alerts permanently muted for chat %d", chat_id)

    def unmute(self, chat_id: int) -> None:
        with self._chat_ids_lock:
            self._muted.discard(chat_id)
            self._save_muted()
        logger.info("Alerts unmuted for chat %d", chat_id)

    def is_muted(self, chat_id: int) -> bool:
        with self._chat_ids_lock:
            return chat_id in self._muted

    def snooze(self, chat_id: int, seconds: float) -> None:
        with self._chat_ids_lock:
            self._snooze_until[chat_id] = time.monotonic() + seconds
        logger.info("Alerts snoozed for chat %d for %.0fs", chat_id, seconds)

    def snooze_reset(self, chat_id: int) -> None:
        with self._chat_ids_lock:
            self._snooze_until.pop(chat_id, None)
        logger.info("Snooze cancelled for chat %d", chat_id)

    def snooze_remaining(self, chat_id: int) -> float:
        with self._chat_ids_lock:
            return max(0.0, self._snooze_until.get(chat_id, 0.0) - time.monotonic())

    def update_chat_ids(self, chat_ids: list[int]) -> None:
        with self._chat_ids_lock:
            self._chat_ids = chat_ids
            logger.info("Telegram chat IDs updated")

    def set_alert_threshold(self, threshold: int) -> None:
        self._alert_threshold = threshold

    def set_alert_cooldown(self, cooldown: float) -> None:
        self._alert_cooldown = cooldown

    def set_escalation_threshold(self, threshold: int) -> None:
        self._escalation_threshold = threshold

    def send_alert(self, score: int, summary: str, description: str, frames: list[np.ndarray], messages: list[dict] | None = None) -> None:
        with self._chat_ids_lock:
            now = time.monotonic()
            chat_ids = [cid for cid in self._chat_ids if cid not in self._muted and self._snooze_until.get(cid, 0.0) <= now]
            thresholds = {cid: self._thresholds.get(cid, self._alert_threshold) for cid in chat_ids}

        should_save = self._save_alerts and score >= self._alert_threshold
        eligible_chats = [cid for cid in chat_ids if score >= thresholds[cid]]

        now_mono = time.monotonic()
        cooldown_expired = (now_mono - self._last_alert_time) >= self._alert_cooldown
        escalated = score >= self._last_alert_score + self._escalation_threshold
        should_send = bool(eligible_chats) and (cooldown_expired or escalated)

        if not should_save and not should_send:
            return

        video_bytes = compile_video(frames, self._video_fps)
        ts = time.strftime("%Y%m%d_%H%M%S")

        if should_save:
            alert_path = self._alerts_dir / f"{ts}_score{score}.mp4"
            try:
                alert_path.write_bytes(video_bytes)
                alert_path.chmod(0o666)
            except OSError:
                logger.exception("Failed to save alert video to %s", alert_path)
            if messages is not None:
                json_path = alert_path.with_suffix(".json")
                try:
                    user_content = next((m["content"] for m in messages if m["role"] == "user"), [])
                    json_path.write_text(json.dumps(user_content, indent=2))
                    json_path.chmod(0o666)
                except OSError:
                    logger.exception("Failed to save alert JSON to %s", json_path)

        if not should_send:
            return

        self._last_alert_time = now_mono
        self._last_alert_score = score
        text = f"{score} - {summary}\n\n{description}"
        reply_markup = json.dumps({"inline_keyboard": [[
            {"text": "Live", "url": self._live_stream_url},
            {"text": "Logs", "url": self._logs_url},
            {"text": "🔕 5m", "callback_data": "snooze:300"},
            {"text": "🔕 15m", "callback_data": "snooze:900"},
        ]]})
        for chat_id in eligible_chats:
            requests.post(
                f"{self._url}/sendVideo",
                data={"chat_id": chat_id, "caption": text, "reply_markup": reply_markup},
                files={"video": ("alert.mp4", video_bytes, "video/mp4")},
                timeout=60,
            ).raise_for_status()

    def send_system_alert(self, description: str) -> None:
        with self._chat_ids_lock:
            now = time.monotonic()
            chat_ids = [
                cid for cid in self._chat_ids
                if cid not in self._muted and cid not in self._sysalert_disabled and self._snooze_until.get(cid, 0.0) <= now
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
                    json={"offset": offset, "timeout": 30, "allowed_updates": ["message", "callback_query"]},
                    timeout=35,
                )
                for update in resp.json().get("result", []):
                    offset = update["update_id"] + 1

                    if "callback_query" in update:
                        cq = update["callback_query"]
                        cq_id = cq["id"]
                        chat_id = cq.get("message", {}).get("chat", {}).get("id")
                        data = cq.get("data", "")
                        with self._chat_ids_lock:
                            allowed = chat_id in self._chat_ids
                        toast = ""
                        if chat_id and allowed and data.startswith("snooze:"):
                            try:
                                seconds = float(data.split(":", 1)[1])
                                self.snooze(chat_id, seconds)
                                mins = int(seconds // 60)
                                toast = f"Snoozed for {mins} minute{'s' if mins != 1 else ''}."
                            except (ValueError, IndexError):
                                pass
                        try:
                            requests.post(
                                f"{self._url}/answerCallbackQuery",
                                json={"callback_query_id": cq_id, "text": toast},
                                timeout=10,
                            )
                        except Exception:
                            logger.exception("Failed to answer callback query")
                        continue

                    msg = update.get("message", {})
                    chat_id = msg.get("chat", {}).get("id")
                    text = msg.get("text", "")
                    key = (text.split()[0] if text.startswith("/") else None) if text else None
                    with self._chat_ids_lock:
                        allowed = chat_id in self._chat_ids
                    if chat_id and key in commands and allowed:
                        try:
                            result = commands[key](chat_id, text)
                            if isinstance(result, tuple) and isinstance(result[1], dict):
                                text_body, reply_markup = result
                                requests.post(
                                    f"{self._url}/sendMessage",
                                    json={"chat_id": chat_id, "text": text_body, "reply_markup": reply_markup},
                                    timeout=10,
                                )
                            elif isinstance(result, tuple):
                                caption, frames = result
                                requests.post(
                                    f"{self._url}/sendMessage",
                                    json={"chat_id": chat_id, "text": caption},
                                    timeout=10,
                                ).raise_for_status()
                                video_bytes = compile_video(frames, self._video_fps)
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
                            logger.exception("Failed to handle %s", key)
            except Exception:
                logger.exception("Poll error")
                time.sleep(5)


