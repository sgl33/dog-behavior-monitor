from dataclasses import dataclass, field
from pathlib import Path

import yaml


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
    dog_name: str
    dog_description: str
    no_detection_fallback_seconds: float
    fallback_detection_enabled: bool
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
        dog_name=raw["dog_name"],
        dog_description=raw["dog_description"],
        no_detection_fallback_seconds=raw["no_detection_fallback_seconds"],
        fallback_detection_enabled=raw.get("fallback_detection_enabled", True),
    )
