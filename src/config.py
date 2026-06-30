"""Config schema. See `sample-config.yaml` for info."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from detector import Detector
    from eval_saver import EvalSaver
    from llm import LLMClient
    from manager import Manager
    from memory_query import MemoryQuerier
    from telegram import TelegramClient

logger = logging.getLogger(__name__)


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
    recovery_seconds: float = 10.0


@dataclass
class LLMEndpointConfig:
    vision_model: str
    vision_url: str
    fast_model: str
    fast_url: str
    memory_model: str
    memory_url: str
    frame_sampling: list[dict]
    detection_window: float
    crop_padding: float
    max_tokens: int
    cooldown: float
    min_interval: float
    slow_threshold: float
    vision_token: str | None = None
    fast_token: str | None = None
    memory_token: str | None = None


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
    save_alerts: bool = True


@dataclass
class DoublePassConfig:
    # When enabled, an LLM result at/above the alert threshold is re-run to
    # confirm before alerting. verify_* set a separate model/endpoint for that
    # second pass; when unset, it reuses the vision model/endpoint.
    enabled: bool = True
    verify_model: str | None = None
    verify_url: str | None = None
    verify_token: str | None = None


@dataclass
class Config:
    streams: dict[str, StreamConfig]
    recorder: RecorderConfig
    llm_endpoint: LLMEndpointConfig
    telegram: TelegramConfig
    web_server: WebServerConfig
    detect_interval: float
    camera_stale_threshold: int
    yolo_source_model: Path
    yolo_device: str
    yolo_image_size: int
    dog_name: str
    dog_description: str
    no_detection_fallback_seconds: float
    fallback_detection_enabled: bool
    eval_cap: int
    alert_cap: int = 1000
    double_pass: DoublePassConfig = field(default_factory=DoublePassConfig)
    healthcheck_url: str | None = None
    video_playback_speed: float = 4.0
    yolo_model_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.yolo_model_path = self.yolo_source_model.parent / f"{self.yolo_source_model.stem}_int8_openvino_model"


def load_config(path: Path) -> Config:
    """
    Load config from YAML file in `path`.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(
        streams={k: StreamConfig(**v) for k, v in raw["streams"].items()},
        recorder=RecorderConfig(**raw["recorder"]),
        llm_endpoint=LLMEndpointConfig(**raw["llm_endpoint"]),
        telegram=TelegramConfig(**raw["telegram"]),
        web_server=WebServerConfig(**raw["web_server"]),
        detect_interval=raw["detect_interval"],
        camera_stale_threshold=raw["camera_stale_threshold"],
        yolo_source_model=path.parent / raw["yolo_source_model"],
        yolo_device=raw["yolo_device"],
        yolo_image_size=raw["yolo_image_size"],
        dog_name=raw["dog_name"],
        dog_description=raw["dog_description"],
        no_detection_fallback_seconds=raw["no_detection_fallback_seconds"],
        fallback_detection_enabled=raw.get("fallback_detection_enabled", True),
        eval_cap=raw.get("eval_cap", 200),
        alert_cap=raw.get("alert_cap", 1000),
        healthcheck_url=raw.get("healthcheck_url"),
        video_playback_speed=raw.get("video_playback_speed", 4.0),
        double_pass=DoublePassConfig(**raw.get("double_pass", {})),
    )


def watch_config(
    path: Path,
    config: Config,
    telegram_client: TelegramClient,
    eval_saver: EvalSaver,
    llm_client: LLMClient,
    memory_querier: MemoryQuerier,
    manager: Manager,
    detectors: dict[str, Detector],
) -> None:
    """
    Watch config file for changes and update relevant components in-place.
    """
    last_mtime = path.stat().st_mtime
    while True:
        time.sleep(5.0)
        try:
            mtime = path.stat().st_mtime
            if mtime == last_mtime:
                continue
            last_mtime = mtime
            new_config = load_config(path)

            tg = new_config.telegram
            telegram_client.update_chat_ids(tg.chat_ids)
            telegram_client.set_alert_threshold(tg.alert_threshold)
            telegram_client.set_alert_cooldown(tg.alert_cooldown)
            telegram_client.set_escalation_threshold(tg.escalation_threshold)
            eval_saver.set_alert_threshold(tg.alert_threshold)
            eval_saver.set_eval_cap(new_config.eval_cap)
            eval_saver.set_alert_cap(new_config.alert_cap)
            logger.info(
                "Reloaded telegram: chat_ids=%s alert_threshold=%s alert_cooldown=%s escalation_threshold=%s",
                tg.chat_ids, tg.alert_threshold, tg.alert_cooldown, tg.escalation_threshold,
            )

            ep = new_config.llm_endpoint
            llm_client.set_vision_model(ep.vision_model)
            llm_client.set_vision_endpoint(ep.vision_url, ep.vision_token)
            dp = new_config.double_pass
            llm_client.set_verify_model(dp.verify_model or ep.vision_model)
            llm_client.set_verify_endpoint(
                dp.verify_url or ep.vision_url,
                dp.verify_token if dp.verify_url else ep.vision_token,
            )
            llm_client.set_fast_model(ep.fast_model)
            llm_client.set_fast_endpoint(ep.fast_url, ep.fast_token)
            llm_client.set_memory_model(ep.memory_model)
            llm_client.set_memory_endpoint(ep.memory_url, ep.memory_token)
            llm_client.set_dog_description(new_config.dog_description)
            llm_client.set_frame_sampling(ep.frame_sampling)
            llm_client.set_crop_padding(ep.crop_padding)
            llm_client.set_max_tokens(ep.max_tokens)
            logger.info(
                "Reloaded llm: vision=%s fast=%s memory=%s detection_window=%s crop_padding=%s max_tokens=%s",
                ep.vision_model, ep.fast_model, ep.memory_model,
                ep.detection_window, ep.crop_padding, ep.max_tokens,
            )

            memory_querier.set_dog_name(new_config.dog_name)

            manager.set_fallback_detection_enabled(new_config.fallback_detection_enabled)
            manager.set_double_pass_enabled(new_config.double_pass.enabled)
            manager.set_cooldown(ep.cooldown)
            manager.set_min_interval(ep.min_interval)
            manager.set_detection_window(ep.detection_window)
            manager.set_slow_threshold(ep.slow_threshold)
            manager.set_no_detection_interval(new_config.no_detection_fallback_seconds)
            logger.info(
                "Reloaded manager: fallback=%s double_pass=%s cooldown=%s min_interval=%s detection_window=%s slow_threshold=%s no_detection_interval=%s",
                new_config.fallback_detection_enabled, new_config.double_pass.enabled, ep.cooldown, ep.min_interval,
                ep.detection_window, ep.slow_threshold, new_config.no_detection_fallback_seconds,
            )

            for det in detectors.values():
                det.set_detect_interval(new_config.detect_interval)
            logger.info("Reloaded detect_interval=%s", new_config.detect_interval)

            config.camera_stale_threshold = new_config.camera_stale_threshold
        except Exception:
            logger.exception("Failed to reload config")
