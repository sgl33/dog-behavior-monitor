import logging
import random
import time
from pathlib import Path

import numpy as np

from utils import save_clip

logger = logging.getLogger(__name__)


class EvalSaver:
    """
    Saves video files and data to `data/alerts` and `data/eval` to be used for
    LLM evaluation purposes.
    """

    def __init__(
        self,
        data_dir: Path,
        alert_threshold: int,
        video_fps: float,
        eval_cap: int = 200,
        save_alerts: bool = True,
        alert_cap: int = 1000,
    ):
        self._alert_threshold = alert_threshold
        self._video_fps = video_fps
        self._eval_cap = eval_cap
        self._save_alerts = save_alerts
        self._alert_cap = alert_cap
        self._eval_dir = data_dir / "eval"
        self._eval_dir.mkdir(exist_ok=True)
        self._eval_dir.chmod(0o777)
        self._alerts_dir = data_dir / "alerts"
        self._alerts_dir.mkdir(exist_ok=True)
        self._alerts_dir.chmod(0o777)

    def set_alert_threshold(self, threshold: int) -> None:
        self._alert_threshold = threshold

    def set_eval_cap(self, cap: int) -> None:
        self._eval_cap = cap

    def set_alert_cap(self, cap: int) -> None:
        self._alert_cap = cap

    def save_alert(
        self, 
        score: int, 
        messages: list[dict], 
        frames: list[np.ndarray]
    ) -> None:
        """
        Save positive examples (at or above threshold) to data/alerts for
        later LLM evaluation, then delete the oldest alerts beyond the cap.
        """
        if not self._save_alerts or score < self._alert_threshold:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_clip(self._alerts_dir / f"{ts}_score{score}", messages, frames, self._video_fps)
        self._rotate_alerts()

    def _rotate_alerts(self) -> None:
        """
        Keep only the newest `_alert_cap` alert clips, deleting the oldest
        (both the .mp4 and .json of each) once the cap is exceeded.
        """
        stems = sorted({
            p.stem for p in self._alerts_dir.iterdir()
            if p.suffix in (".mp4", ".json")
        })
        for stem in stems[:len(stems) - self._alert_cap]:
            for suffix in (".mp4", ".json"):
                try:
                    (self._alerts_dir / f"{stem}{suffix}").unlink(missing_ok=True)
                except OSError:
                    logger.exception("Failed to delete old alert %s%s", stem, suffix)

    def save_negative(
        self,
        score: int,
        messages: list[dict],
        frames: list[np.ndarray],
        chance: float = 0.05
    ) -> None:
        """
        Save negative examples (under threshold) for LLM evaluation at a
        random chance.
        """
        if score >= self._alert_threshold:
            return
        existing = sum(
            1 for p in self._eval_dir.iterdir() 
            if p.suffix == ".json"
        )
        if existing >= self._eval_cap or random.random() >= chance:
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        save_clip(self._eval_dir / f"{ts}_score{score}", messages, frames, self._video_fps)
