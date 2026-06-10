"""
Handlers for Telegram commands.
"""
from collections.abc import Callable
from datetime import datetime

from config import Config
from utils import format_age
from llm_logger import LLMOutputLogger
from manager import Manager
from recorder import Recorder
from telegram import TelegramClient
from web_server import WebServerClient

CommandMap = dict[
    str | None,
    Callable[
        [int, str],
        str | tuple[str, list]
    ]
]


def build_commands(
    telegram_client: TelegramClient,
    manager: Manager,
    recorders: dict[str, Recorder],
    web_client: WebServerClient,
    config: Config,
    llm_logger: LLMOutputLogger,
) -> CommandMap:
    """
    Returns a `CommandMap` object mapping Telegram commands to handler 
    functions.
    """
    camera_stale_threshold = config.camera_stale_threshold
    live_stream_url = config.telegram.live_stream_url

    def status_fn(_chat_id: int, _text: str) -> str:
        """
        Handler for `/status` command.
        - `/status`: shows the status of all cameras and the most recent LLM 
            inference time
        """
        now = datetime.now()
        lines = ["📷 Cameras:"]

        for camera, recorder in recorders.items():
            ts = recorder.last_frame_time()
            if ts is None:
                lines.append(f"⚫ {camera}: no frames yet")
            else:
                age = (now - ts).total_seconds()
                icon = "🟢" if age < camera_stale_threshold else "🔴"
                lines.append(f"{icon} {camera}: last frame {age:.0f}s ago")
        latency = manager.last_llm_inference_latency

        lines.append("")
        llm_state = "on" if manager.llm_enabled else "off"
        lines.append(f"🧠 LLM inference: {llm_state}")
        lines.append(
            f"⏱ Most recent LLM inference time: {latency:.1f}s"
            if latency is not None else "⏱ No LLM inference yet"
        )
        return "\n".join(lines)

    def last_fn(_chat_id: int, _text: str) -> str | tuple[str, list]:
        """
        Handler for `/last` command.
        - `/last`: shows the most recent LLM output
        """
        result = manager.last_result
        if result is None:
            return "No LLM output yet."
        score, description, ts = result

        caption = (
            f"[{format_age(ts)}] {score}: {description} "
            f"(took {manager._last_llm_inference_latency:.2f} sec)"
        )
        frames = manager.last_frames
        if frames:
            return caption, frames
        return caption

    def score_fn(chat_id: int, text: str) -> str:
        """
        Handler for `/score [0-10]` command.
        - `/score`: shows current alert threshold
        - `/score [0-10]`: sets alert threshold to the specified value
        """
        parts = text.split()
        if len(parts) == 1:
            return (
                f"Your alert threshold is {telegram_client.get_threshold(chat_id)}."
                " Run /score [0-10] to change it."
            )
        elif len(parts) > 2:
            return "Usage: /score [0-10]"

        try:
            threshold = int(parts[1])
        except ValueError:
            return "Usage: /score [0-10]"
        if not (0 <= threshold <= 10):
            return "Threshold must be between 0 and 10."
        telegram_client.set_threshold(chat_id, threshold)
        return f"Your alert threshold is now {threshold}."

    def sysalert_fn(chat_id: int, text: str) -> str:
        """
        Handler for `/sysalert [on|off]` command.
        - `/sysalert`: shows whether system alerts are on or off
        - `/sysalert on`: turns on system alerts for the user
        - `/sysalert off`: turns off system alerts for the user

        System alerts include camera online/offline alerts, LLM errors, and
        other notifications not related to dog behavior detection.
        """
        parts = text.split()
        if len(parts) == 1:
            enabled = telegram_client.get_sysalert(chat_id)
            return f"System alerts are {'on' if enabled else 'off'} for you. Run /sysalert {'off' if enabled else 'on'} to change."
        if len(parts) != 2 or parts[1] not in ("on", "off"):
            return "Usage: /sysalert [on|off]"
        enabled = parts[1] == "on"
        telegram_client.set_sysalert(chat_id, enabled)
        return f"System alerts turned {parts[1]}."

    def mute_fn(chat_id: int, text: str) -> str:
        """
        Handler for `/mute [#h|#m]` command.
        - `/mute`: shows remaining mute time
        - `/mute #h`: mutes alerts for the specified number of hours
        - `/mute #m`: mutes alerts for the specified number of minutes
        """
        remaining = telegram_client.mute_remaining(chat_id)
        parts = text.split()
        if len(parts) == 1:
            if remaining > 0:
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                return f"Alerts are muted for the next {mins} minutes {secs} seconds. Run /unmute to cancel."
            return "Alerts are not muted. Run /mute #h (hours) or /mute #m (minutes) to mute."
        if len(parts) != 2:
            return "Usage: /mute [{hrs}h|{mins}m] (example: `/mute 2h`, `/mute 30m`)"

        time_str = parts[1]
        try:
            if time_str.endswith("h"):
                seconds = float(time_str[:-1]) * 3600
            elif time_str.endswith("m"):
                seconds = float(time_str[:-1]) * 60
            else:
                return "Usage: /mute [{hrs}h|{mins}m]"
        except ValueError:
            return "Usage: /mute [{hrs}h|{mins}m]"

        telegram_client.mute(chat_id, seconds)
        return f"Alerts muted for {time_str}."

    def unmute_fn(chat_id: int, _text: str) -> str:
        """
        Handler for `/unmute` command.
        - `/unmute`: unmutes alerts immediately
        """
        if telegram_client.mute_remaining(chat_id) == 0:
            return "Alerts are not muted."
        telegram_client.unmute(chat_id)
        return "Alerts unmuted."

    def logs_fn(_chat_id: int, _text: str) -> str:
        """
        Handler for `/logs` command.
        - `/logs`: shows the URL to the live LLM feed
        """
        return f"📊 Live LLM feed:\n{web_client.public_url}"

    def live_fn(_chat_id: int, _text: str) -> str:
        """
        Handler for `/live` command.
        - `/live`: shows the URL to the live stream
        """
        return f"📹 Live stream:\n{live_stream_url}"

    def llm_fn(_chat_id: int, text: str) -> str:
        """
        Handler for `/llm [on|off]` command.
        - `/llm`: shows whether LLM inference is enabled
        - `/llm on`: enables LLM inference
        - `/llm off`: disables LLM inference
        """
        parts = text.split()
        if len(parts) == 1:
            state = "on" if manager.llm_enabled else "off"
            return f"LLM inference is {state}. Run /llm {'off' if manager.llm_enabled else 'on'} to change."
        if len(parts) != 2 or parts[1] not in ("on", "off"):
            return "Usage: /llm [on|off]"
        enabled = parts[1] == "on"
        manager.set_llm_enabled(enabled)
        return f"LLM inference turned {parts[1]}."

    def ask_fn(chat_id: int, text: str) -> str:
        """
        Handler for `/ask <question>` command.
        - `/ask <question>`: asks a question to the LLM
        """
        parts = text.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            return "Usage: /ask <question>"
        try:
            return llm_logger.query(parts[1].strip(), chat_id=chat_id)
        except Exception as e:
            return f"❌ Error: {e}"

    def catchall_fn(chat_id: int, text: str) -> str:
        """
        Handler for general text not starting with a recognized command. 
        Forwards the text to the LLM logger as a query.
        """
        if not text.strip():
            return "Usage: /ask <question>"
        try:
            return llm_logger.query(text.strip(), chat_id=chat_id)
        except Exception as e:
            return f"❌ Error: {e}"

    return {
        "/status": status_fn,
        "/last": last_fn,
        "/logs": logs_fn,
        "/live": live_fn,
        "/score": score_fn,
        "/sysalert": sysalert_fn,
        "/mute": mute_fn,
        "/unmute": unmute_fn,
        "/llm": llm_fn,
        "/ask": ask_fn,
        None: catchall_fn,
    }
