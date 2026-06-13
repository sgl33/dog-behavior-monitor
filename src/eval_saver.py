import json
import logging
import random
import time
from pathlib import Path

import numpy as np

from utils import compile_video

logger = logging.getLogger(__name__)


class EvalSaver:
    def __init__(self, data_dir: Path, alert_threshold: int, video_fps: float, eval_cap: int = 200):
        self._alert_threshold = alert_threshold
        self._video_fps = video_fps
        self._eval_cap = eval_cap
        self._eval_dir = data_dir / "eval"
        self._eval_dir.mkdir(exist_ok=True)
        self._eval_dir.chmod(0o777)

    def set_alert_threshold(self, threshold: int) -> None:
        self._alert_threshold = threshold

    def set_eval_cap(self, cap: int) -> None:
        self._eval_cap = cap

    def maybe_save(self, score: int, messages: list[dict], frames: list[np.ndarray]) -> None:
        if score >= self._alert_threshold:
            return
        existing = sum(1 for p in self._eval_dir.iterdir() if p.suffix == ".json")
        if existing >= self._eval_cap:
            return
        if random.random() >= 0.05:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = self._eval_dir / f"{ts}_score{score}"
        user_content = next((m["content"] for m in messages if m["role"] == "user"), [])
        try:
            p = base.with_suffix(".json")
            p.write_text(json.dumps(user_content, indent=2))
            p.chmod(0o666)
        except OSError:
            logger.exception("Failed to save eval JSON to %s", base)
        try:
            video_bytes = compile_video(frames, self._video_fps)
            p = base.with_suffix(".mp4")
            p.write_bytes(video_bytes)
            p.chmod(0o666)
        except Exception:
            logger.exception("Failed to save eval video to %s", base)
