import logging
import os
import signal
import time
from pathlib import Path
import threading

import numpy as np
import yaml
from ultralytics import YOLO

from telegram_commands import build_commands
from config import Config, load_config, watch_config
from eval_saver import EvalSaver
from llm_logger import LLMOutputLogger
from detector import Detector, YoloLagMonitor
from llm import LLMClient
from manager import Manager
from memory_query import MemoryQuerier
from recorder import Recorder
from state import DogDetectionState
from telegram import TelegramClient
from utils import ColorFormatter, healthcheck_ping
from web_server import WebServerClient

logger = logging.getLogger(__name__)


def ensure_model_exported(config: Config) -> None:
    """
    If using an Intel iGPU, ensure the YOLO model is exported to OpenVINO 
    format with the correct image size. This is idempotent; nothing will happen
    if the model is already exported.
    """
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


def setup_logging() -> None:
    """Configure the root logger with the colorized formatter."""
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    logging.basicConfig(level=level, handlers=[handler])
    for noisy in ("urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def compute_video_fps(
    config: Config,
    default: float = 5.0,
    multiplier: float = 4.0
) -> float:
    """Derive the output video FPS from the LLM frame-sampling tiers."""
    tiers = config.llm_endpoint.frame_sampling
    total_frames = sum(round(t["fps"] * t["seconds"]) for t in tiers)
    total_seconds = sum(t["seconds"] for t in tiers)
    return (
        total_frames / total_seconds 
        if total_seconds > 0 else default
    ) * multiplier


def init_yolo(config: Config) -> YOLO:
    """Load (exporting first if needed) and warm up the YOLO model."""
    if config.yolo_device.startswith("intel"):
        ensure_model_exported(config)
        model = YOLO(config.yolo_model_path)
    else:
        model = YOLO(config.yolo_source_model)
    model.predict(
        np.zeros((config.yolo_image_size, config.yolo_image_size, 3), dtype=np.uint8),
        device=config.yolo_device,
        imgsz=config.yolo_image_size,
        half=True,
        verbose=False,
    )
    return model


def main():
    """
    Main function of the application, performing initialization and object and
    thread instantiation.
    """
    setup_logging()

    # Load config / initialize state
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(config_path)
    video_fps = compute_video_fps(config)
    cameras = list(config.streams.keys())
    state = DogDetectionState(cameras)

    # Initialize YOLO
    model = init_yolo(config)
    model_lock = threading.Lock()

    # Instantiate objects
    web_client = WebServerClient(config.web_server)
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
    llm_client = LLMClient(
        config=config.llm_endpoint, 
        dog_description=config.dog_description
    )
    llm_logger = LLMOutputLogger(
        data_dir=Path(__file__).parent.parent / "data",
        llm_client=llm_client,
    )
    memory_querier = MemoryQuerier(
        store=llm_logger,
        llm_client=llm_client,
        dog_name=config.dog_name,
    )
    eval_saver = EvalSaver(
        data_dir=Path(__file__).parent.parent / "data",
        alert_threshold=config.telegram.alert_threshold,
        video_fps=video_fps,
        eval_cap=config.eval_cap,
        save_alerts=config.telegram.save_alerts,
        alert_cap=config.alert_cap,
    )
    lag_monitor = YoloLagMonitor(
        detect_interval=config.detect_interval,
        telegram_client=telegram_client,
    )
    detectors = {
        camera: Detector(
            camera_name=camera,
            recorder=recorders[camera],
            state=state,
            model=model,
            model_lock=model_lock,
            telegram_client=telegram_client,
            config=config,
            lag_monitor=lag_monitor,
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
        eval_saver=eval_saver,
    )

    # Instantiate/start threads
    threading.Thread(
        target=watch_config,
        args=(config_path, config, telegram_client, eval_saver, llm_client, memory_querier, manager, detectors),
        daemon=True,
        name="config-watcher",
    ).start()
    threading.Thread(
        target=web_client.run_camera_status_loop,
        args=(recorders, config),
        daemon=True,
        name="camera-status",
    ).start()
    if config.healthcheck_url:
        threading.Thread(
            target=healthcheck_ping,
            args=(config.healthcheck_url,),
            daemon=True,
            name="healthcheck",
        ).start()

    telegram_client.start_polling(build_commands(
        telegram_client=telegram_client,
        manager=manager,
        recorders=recorders,
        config=config,
        memory_querier=memory_querier,
        lag_monitor=lag_monitor,
    ))

    # Graceful shutdown: on SIGTERM/SIGINT, signal every loop to stop so the
    # main thread's manager.join() returns and the process exits promptly.
    # Without this, the process (PID 1 under tini, or PID 1 directly) ignores
    # SIGTERM and docker has to wait out the grace period before SIGKILL.
    def _shutdown(signum, _frame):
        logger.info("Received signal %s, shutting down", signum)
        manager.stop()
        for d in detectors.values():
            d.stop()
        for r in recorders.values():
            r.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Stagger recorder startup to avoid overwhelming the system
    for r in recorders.values():
        r.start()
        time.sleep(0.5)
    for d in detectors.values():
        d.start()
    manager.start()

    manager.join()


if __name__ == "__main__":
    main()