from collections.abc import Callable
from datetime import datetime

from manager import Manager
from recorder import Recorder
from telegram import TelegramClient
from web_server import WebServerClient

CommandMap = dict[
    str, 
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
    camera_stale_threshold: int,
    live_stream_url: str,
) -> CommandMap:
    def status_fn(_chat_id: int, _text: str) -> str:
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
        lines.append(
            f"⏱ Most recent LLM inference time: {latency:.1f}s" 
            if latency is not None else "⏱ No LLM inference yet"
        )
        return "\n".join(lines)

    def last_fn(_chat_id: int, _text: str) -> str | tuple[str, list]:
        result = manager.last_result
        if result is None:
            return "No LLM output yet."
        score, description, ts = result
        age = (datetime.now() - ts).total_seconds()
        if age < 60:
            age_str = f"{age:.0f}s ago"
        elif age < 3600:
            age_str = f"{age / 60:.0f}m ago"
        else:
            age_str = f"{age / 3600:.1f}h ago"
        caption = f"[{age_str}] {score}: {description} (took {manager._last_llm_inference_latency:.2f} sec)"
        frames = manager.last_frames
        if frames:
            return caption, frames
        return caption

    def score_fn(chat_id: int, text: str) -> str:
        parts = text.split()
        if len(parts) == 1:
            return f"Your alert threshold is {telegram_client.get_threshold(chat_id)}. Run /score [0-10] to change it."
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
        remaining = telegram_client.mute_remaining(chat_id)
        parts = text.split()
        if len(parts) == 1:
            if remaining > 0:
                mins = int(remaining // 60)
                secs = int(remaining % 60)
                return f"Alerts are muted for the next {mins} minutes {secs} seconds. Run /unmute to cancel."
            return "Alerts are not muted. Run /mute [{hrs}h|{mins}m] to mute."
        if len(parts) != 2:
            return "Usage: /mute [{hrs}h|{mins}m] (example: `/mute 2h`, `/mute 30m`)"
        arg = parts[1]
        try:
            if arg.endswith("h"):
                seconds = float(arg[:-1]) * 3600
            elif arg.endswith("m"):
                seconds = float(arg[:-1]) * 60
            else:
                return "Usage: /mute [{hrs}h|{mins}m]"
        except ValueError:
            return "Usage: /mute [{hrs}h|{mins}m]"
        telegram_client.mute(chat_id, seconds)
        return f"Alerts muted for {arg}."

    def unmute_fn(chat_id: int, _text: str) -> str:
        if telegram_client.mute_remaining(chat_id) == 0:
            return "Alerts are not muted."
        telegram_client.unmute(chat_id)
        return "Alerts unmuted."

    def logs_fn(_chat_id: int, _text: str) -> str:
        return f"📊 Live LLM feed:\n{web_client.public_url}"

    def live_fn(_chat_id: int, _text: str) -> str:
        return f"📹 Live stream:\n{live_stream_url}"

    return {
        "/status": status_fn,
        "/last": last_fn,
        "/logs": logs_fn,
        "/live": live_fn,
        "/score": score_fn,
        "/sysalert": sysalert_fn,
        "/mute": mute_fn,
        "/unmute": unmute_fn,
    }
