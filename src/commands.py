"""
Handlers for Telegram commands.
"""
from collections.abc import Callable
from datetime import datetime

from config import Config
from llm_logger import LLMOutputLogger
from manager import Manager
from recorder import Recorder
from telegram import TelegramClient

CommandMap = dict[
    str | None,
    Callable[
        [int, str],
        str | tuple[str, list] | tuple[str, dict]
    ]
]


def build_commands(
    telegram_client: TelegramClient,
    manager: Manager,
    recorders: dict[str, Recorder],
    config: Config,
    llm_logger: LLMOutputLogger,
) -> CommandMap:
    """
    Returns a `CommandMap` object mapping Telegram commands to handler 
    functions.
    """
    camera_stale_threshold = config.camera_stale_threshold
    live_stream_url = config.telegram.live_stream_url
    logs_url = config.telegram.logs_url

    def status_fn(_chat_id: int, _text: str) -> tuple[str, dict]:
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
        last_infer_time = manager.last_llm_finish_wall_time

        lines.append("")
        llm_state = "on" if manager.llm_enabled else "off"
        lines.append(f"🧠 LLM inference: {llm_state}")
        if latency is not None and last_infer_time is not None:
            age = (now - last_infer_time.replace(tzinfo=None)).total_seconds()
            lines.append(f"⏱ Most recent LLM inference: {latency:.1f}s, {age:.0f}s ago")
        else:
            lines.append("⏱ No LLM inference yet")
        reply_markup = {"inline_keyboard": [[
            {"text": "Live Stream", "url": live_stream_url},
            {"text": "Logs", "url": logs_url},
        ]]}
        return "\n".join(lines), reply_markup

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

    def snooze_fn(chat_id: int, text: str) -> str:
        """
        Handler for `/snooze [#h|#m|reset]` command.
        - `/snooze`: shows remaining snooze time
        - `/snooze #h`: snoozes alerts for the specified number of hours
        - `/snooze #m`: snoozes alerts for the specified number of minutes
        - `/snooze reset`: cancels the active snooze immediately
        """
        remaining = telegram_client.snooze_remaining(chat_id)
        parts = text.split()
        if len(parts) == 1:
            if remaining > 0:
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                return f"Alerts snoozed for the next {mins} minutes {secs} seconds. Run /snooze reset to cancel."
            return "Alerts are not snoozed. Run /snooze #h (hours) or /snooze #m (minutes) to snooze."
        if len(parts) != 2:
            return "Usage: /snooze [{hrs}h|{mins}m|reset] (example: `/snooze 2h`, `/snooze 30m`)"

        arg = parts[1]
        if arg == "reset":
            if remaining == 0:
                return "Alerts are not snoozed."
            telegram_client.snooze_reset(chat_id)
            return "Snooze cancelled."

        try:
            if arg.endswith("h"):
                seconds = float(arg[:-1]) * 3600
            elif arg.endswith("m"):
                seconds = float(arg[:-1]) * 60
            else:
                return "Usage: /snooze [{hrs}h|{mins}m|reset]"
        except ValueError:
            return "Usage: /snooze [{hrs}h|{mins}m|reset]"

        telegram_client.snooze(chat_id, seconds)
        return f"Alerts snoozed for {arg}."

    def mute_fn(chat_id: int, _text: str) -> str:
        """
        Handler for `/mute` command.
        - `/mute`: permanently mutes alerts until /unmute is run
        """
        if telegram_client.is_muted(chat_id):
            return "Alerts are already muted. Run /unmute to resume."
        telegram_client.mute(chat_id)
        return "Alerts muted. Run /unmute to resume."

    def unmute_fn(chat_id: int, _text: str) -> str:
        """
        Handler for `/unmute` command.
        - `/unmute`: resumes alerts after /mute
        """
        if not telegram_client.is_muted(chat_id):
            return "Alerts are not muted."
        telegram_client.unmute(chat_id)
        return "Alerts unmuted."

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
            inverse_state = "off" if manager.llm_enabled else "on"
            return (f"LLM inference is {state}. Run /llm {inverse_state} to change.")
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
        "/score": score_fn,
        "/sysalert": sysalert_fn,
        "/mute": mute_fn,
        "/unmute": unmute_fn,
        "/snooze": snooze_fn,
        "/llm": llm_fn,
        "/ask": ask_fn,
        None: catchall_fn,
    }
