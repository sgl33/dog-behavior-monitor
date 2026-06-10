import logging
import os
import time
from datetime import datetime
from pathlib import Path
import threading

import numpy as np
import yaml
from ultralytics import YOLO

from commands import build_commands
from config import Config, load_config
from llm_logger import LLMOutputLogger
from detector import Detector
from llm import LLMClient
from manager import Manager
from recorder import Recorder
from state import DogDetectionState
from telegram import TelegramClient
from web_server import WebServerClient


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

    web_client = WebServerClient(config.web_server)

    _tiers = config.llm_endpoint.frame_sampling
    _total_frames = sum(round(t["fps"] * t["seconds"]) for t in _tiers)
    _total_seconds = sum(t["seconds"] for t in _tiers)
    video_fps = _total_frames / _total_seconds if _total_seconds > 0 else 5.0
    telegram_client = TelegramClient(
        config=config.telegram,
        video_fps=video_fps,
        data_dir=Path(__file__).parent.parent / "data",
    )
    recorders = {
        camera: Recorder(
            camera=camera,
            rtsp_url=stream.rtsp,
            telegram_client=telegram_client,
            config=config.recorder,
        )
        for camera, stream in config.streams.items()
    }
    llm_client = LLMClient(config=config.llm_endpoint, dog_description=config.dog_description)
    llm_logger = LLMOutputLogger(
        data_dir=Path(__file__).parent.parent / "data",
        llm_client=llm_client,
        dog_name=config.dog_name,
    )
    detectors = {
        camera: Detector(
            camera=camera,
            recorder=recorders[camera],
            state=state,
            model=model,
            model_lock=model_lock,
            telegram_client=telegram_client,
            config=config,
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
        config=config,
        llm_logger=llm_logger,
    )

    def _watch_config(path: Path) -> None:
        last_mtime = path.stat().st_mtime
        while True:
            time.sleep(5.0)
            try:
                mtime = path.stat().st_mtime
                if mtime == last_mtime:
                    continue
                last_mtime = mtime
                new_config = load_config(path)
                telegram_client.update_chat_ids(new_config.telegram.chat_ids)
                logger.info("Reloaded chat_ids from config: %s", new_config.telegram.chat_ids)
                ep = new_config.llm_endpoint
                llm_client.set_vision_model(ep.vision_model)
                llm_client.set_vision_endpoint(ep.vision_url, ep.vision_token)
                llm_client.set_fast_model(ep.fast_model)
                llm_client.set_fast_endpoint(ep.fast_url, ep.fast_token)
                llm_client.set_memory_model(ep.memory_model)
                llm_client.set_memory_endpoint(ep.memory_url, ep.memory_token)
                logger.info("Reloaded models: vision=%s fast=%s memory=%s", ep.vision_model, ep.fast_model, ep.memory_model)
            except Exception:
                logger.exception("Failed to reload config")

    def _push_camera_status() -> None:
        while True:
            time.sleep(3)
            try:
                now = datetime.now()
                statuses = {
                    cam: (
                        (now - rec.last_frame_time()).total_seconds() <= config.camera_stale_threshold
                        if rec.last_frame_time() else False
                    )
                    for cam, rec in recorders.items()
                }
                web_client.push_camera_status(statuses)
            except Exception:
                logger.exception("Failed to push camera status")

    config_path = Path(__file__).parent.parent / "config.yaml"
    threading.Thread(target=_watch_config, args=(config_path,), daemon=True, name="config-watcher").start()
    threading.Thread(target=_push_camera_status, daemon=True, name="camera-status").start()

    telegram_client.start_polling(build_commands(
        telegram_client=telegram_client,
        manager=manager,
        recorders=recorders,
        web_client=web_client,
        config=config,
        llm_logger=llm_logger,
    ))

    for r in recorders.values():
        r.start()
    for d in detectors.values():
        d.start()
    manager.start()

    manager.join()


if __name__ == "__main__":
    main()