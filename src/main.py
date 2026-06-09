import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

import threading

import numpy as np
from ultralytics import YOLO

from detector import Detector
from llm import LLMClient
from manager import Manager
from recorder import Recorder
from state import DogDetectionState
from telegram import TelegramClient
from web_server import WebServerClient


@dataclass
class StreamConfig:
    name: str
    rtsp: str


@dataclass
class RecorderConfig:
    fps: int
    buffer_seconds: int
    offline_alert_seconds: float
    stale_stream_seconds: float


@dataclass
class LLMEndpointConfig:
    openai_compatible_url: str
    model: str
    frames_per_camera: int
    detection_window: float
    crop_padding: float
    max_tokens: int
    cooldown: float
    slow_threshold: float


@dataclass
class WebServerConfig:
    push_url: str
    public_url: str


@dataclass
class TelegramConfig:
    bot_token: str
    chat_ids: list[int]
    alert_threshold: int
    alert_cooldown: float
    escalation_threshold: int
    live_stream_url: str
    logs_url: str


@dataclass
class Config:
    streams: dict[str, StreamConfig]
    recorder: RecorderConfig
    llm_endpoint: LLMEndpointConfig
    telegram: TelegramConfig
    web_server: WebServerConfig
    detect_interval: float
    manager_loop_interval: float
    camera_stale_threshold: int
    yolo_source_model: Path
    yolo_device: str
    yolo_image_size: int
    dog_description: str
    no_detection_fallback_seconds: float
    yolo_model_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.yolo_model_path = self.yolo_source_model.parent / f"{self.yolo_source_model.stem}_int8_openvino_model"


def load_config(path: Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(
        streams={k: StreamConfig(**v) for k, v in raw["streams"].items()},
        recorder=RecorderConfig(**raw["recorder"]),
        llm_endpoint=LLMEndpointConfig(**raw["llm_endpoint"]),
        telegram=TelegramConfig(**raw["telegram"]),
        web_server=WebServerConfig(**raw["web_server"]),
        detect_interval=raw["detect_interval"],
        manager_loop_interval=raw["manager_loop_interval"],
        camera_stale_threshold=raw["camera_stale_threshold"],
        yolo_source_model=path.parent / raw["yolo_source_model"],
        yolo_device=raw["yolo_device"],
        yolo_image_size=raw["yolo_image_size"],
        dog_description=raw["dog_description"],
        no_detection_fallback_seconds=raw["no_detection_fallback_seconds"],
    )


def ensure_model_exported(config: Config) -> None:
    metadata_path = config.yolo_model_path / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = yaml.safe_load(f)
        if meta.get("imgsz", [0])[0] == config.yolo_image_size:
            return
    logger.info("Exporting model at imgsz=%d...", config.yolo_image_size)
    YOLO(config.yolo_source_model).export(
        format="openvino",
        imgsz=config.yolo_image_size,
        int8=True,
        data="coco8.yaml",
    )
    logger.info("Export complete.")


_LEVEL_COLORS = {
    logging.DEBUG:    "\033[90m",   # gray
    logging.INFO:     "\033[97m",   # white
    logging.WARNING:  "\033[33m",   # yellow
    logging.ERROR:    "\033[31m",   # red
    logging.CRITICAL: "\033[35m",   # magenta
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname:<8}{_RESET}"
        return super().format(record)


logger = logging.getLogger(__name__)


def main():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    handler.setFormatter(_ColorFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    logging.basicConfig(level=level, handlers=[handler])
    config = load_config(Path(__file__).parent.parent / "config.yaml")
    cameras = list(config.streams.keys())
    state = DogDetectionState(cameras)

    ensure_model_exported(config)
    model = YOLO(config.yolo_model_path)
    model.predict(
        np.zeros((config.yolo_image_size, config.yolo_image_size, 3), dtype=np.uint8),
        device=config.yolo_device,
        imgsz=config.yolo_image_size,
        verbose=False,
    )
    model_lock = threading.Lock()

    web_client = WebServerClient(
        push_url=config.web_server.push_url,
        public_url=config.web_server.public_url,
    )

    video_fps = config.llm_endpoint.frames_per_camera / config.llm_endpoint.detection_window
    telegram_client = TelegramClient(
        bot_token=config.telegram.bot_token,
        chat_ids=config.telegram.chat_ids,
        alert_threshold=config.telegram.alert_threshold,
        alert_cooldown=config.telegram.alert_cooldown,
        escalation_threshold=config.telegram.escalation_threshold,
        live_stream_url=config.telegram.live_stream_url,
        logs_url=config.telegram.logs_url,
        video_fps=video_fps,
        data_dir=Path(__file__).parent.parent / "data",
    )
    recorders = {
        camera: Recorder(
            camera=camera,
            rtsp_url=stream.rtsp,
            telegram_client=telegram_client,
            fps=config.recorder.fps,
            buffer_seconds=config.recorder.buffer_seconds,
            offline_alert_seconds=config.recorder.offline_alert_seconds,
            stale_stream_seconds=config.recorder.stale_stream_seconds,
        )
        for camera, stream in config.streams.items()
    }
    llm_client = LLMClient(
        base_url=config.llm_endpoint.openai_compatible_url,
        model=config.llm_endpoint.model,
        dog_description=config.dog_description,
        frames_per_camera=config.llm_endpoint.frames_per_camera,
        crop_padding=config.llm_endpoint.crop_padding,
        max_tokens=config.llm_endpoint.max_tokens,
    )
    detectors = {
        camera: Detector(
            camera=camera,
            recorder=recorders[camera],
            state=state,
            detect_interval=config.detect_interval,
            model=model,
            model_lock=model_lock,
            device=config.yolo_device,
            image_size=config.yolo_image_size,
            telegram_client=telegram_client,
        )
        for camera in cameras
    }
    manager = Manager(
        cameras=cameras,
        state=state,
        recorders=recorders,
        llm_client=llm_client,
        telegram_client=telegram_client,
        web_server=web_client,
        detection_window=config.llm_endpoint.detection_window,
        llm_cooldown=config.llm_endpoint.cooldown,
        loop_interval=config.manager_loop_interval,
        slow_threshold=config.llm_endpoint.slow_threshold,
        no_detection_interval=config.no_detection_fallback_seconds,
    )

    def status_fn(_chat_id: int, _text: str) -> str:
        now = datetime.now()
        lines = ["📷 Cameras:"]
        for camera, recorder in recorders.items():
            ts = recorder.last_frame_time()
            if ts is None:
                lines.append(f"• {camera}: ⚫ no frames yet")
            else:
                age = (now - ts).total_seconds()
                icon = "🟢" if age < config.camera_stale_threshold else "🔴"
                lines.append(f"• {camera}: {icon} last frame {age:.0f}s ago")
        latency = manager.last_llm_inference_latency
        lines.append("")
        lines.append(f"⏱ LLM latency: {latency:.1f}s" if latency is not None else "⏱ LLM latency: N/A")
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

    def _watch_config(path: Path) -> None:
        last_mtime = path.stat().st_mtime
        while True:
            time.sleep(5.0)
            try:
                mtime = path.stat().st_mtime
                if mtime == last_mtime:
                    continue
                last_mtime = mtime
                new_chat_ids = load_config(path).telegram.chat_ids
                telegram_client.update_chat_ids(new_chat_ids)
                logger.info("Reloaded chat_ids from config: %s", new_chat_ids)
            except Exception:
                logger.exception("Failed to reload config")

    config_path = Path(__file__).parent.parent / "config.yaml"
    threading.Thread(target=_watch_config, args=(config_path,), daemon=True, name="config-watcher").start()

    def logs_fn(_chat_id: int, _text: str) -> str:
        return f"📊 Live LLM feed:\n{web_client.public_url}"

    telegram_client.start_polling({
        "/status": status_fn,
        "/last": last_fn,
        "/now": lambda _cid, _txt: manager.trigger_now(),
        "/logs": logs_fn,
        "/score": score_fn,
    })

    for r in recorders.values():
        r.start()
    for d in detectors.values():
        d.start()
    manager.start()

    manager.join()


if __name__ == "__main__":
    main()