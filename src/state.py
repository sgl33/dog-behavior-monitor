from datetime import datetime
from threading import RLock


class DogDetectionState:
    def __init__(self, cameras: list[str]):
        self._detected: dict[str, datetime | None] = {c: None for c in cameras}
        self._lock = RLock()

    def update(self, camera: str) -> None:
        with self._lock:
            self._detected[camera] = datetime.now()

    def get(self, camera: str) -> datetime | None:
        with self._lock:
            return self._detected[camera]

    def any_recent(self, within_seconds: float) -> bool:
        now = datetime.now()
        with self._lock:
            return any(
                ts is not None and (now - ts).total_seconds() <= within_seconds
                for ts in self._detected.values()
            )

    def recent_cameras(self, within_seconds: float) -> list[str]:
        now = datetime.now()
        with self._lock:
            return [
                c for c, ts in self._detected.items()
                if ts is not None and (now - ts).total_seconds() <= within_seconds
            ]
