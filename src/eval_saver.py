import json
import logging
import random
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_EVAL_CAP = 200


class EvalSaver:
    def __init__(self, data_dir: Path, alert_threshold: int):
        self._alert_threshold = alert_threshold
        self._eval_dir = data_dir / "eval"
        self._eval_dir.mkdir(exist_ok=True)

    def set_alert_threshold(self, threshold: int) -> None:
        self._alert_threshold = threshold

    def maybe_save(self, score: int, messages: list[dict]) -> None:
        if score >= self._alert_threshold:
            return
        existing = sum(1 for p in self._eval_dir.iterdir() if p.suffix == ".json")
        if existing >= _EVAL_CAP:
            return
        if random.random() >= 0.5:
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self._eval_dir / f"{ts}_score{score}.json"
        user_content = next((m["content"] for m in messages if m["role"] == "user"), [])
        try:
            path.write_text(json.dumps(user_content, indent=2))
        except OSError:
            logger.exception("Failed to save eval sample to %s", path)
