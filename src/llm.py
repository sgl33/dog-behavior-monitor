import base64
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import requests

from config import LLMEndpointConfig

_JPEG_QUALITY = 85
_LLM_MAX_WIDTH = 640
_LLM_MAX_HEIGHT = 360
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_ANALYZE_PROMPT_PATH = _PROMPTS_DIR / "analyze_prompt.txt"
_DETECT_PROMPT_PATH = _PROMPTS_DIR / "detect_prompt.txt"


class LLMClient:
    def __init__(self, config: LLMEndpointConfig, dog_description: str):
        self._url = f"{config.openai_compatible_url.rstrip('/')}/chat/completions"
        self._model = config.model
        self._dog_description = dog_description
        self._frame_sampling = [(t["seconds"], t["fps"]) for t in config.frame_sampling]
        self._crop_padding = config.crop_padding
        self._max_tokens = config.max_tokens
        self._token = config.token
        self._headers = {"Authorization": f"Bearer {config.token}"} if config.token else {}
        self._set_query_endpoint(config.query_url, config.query_token)

    def set_model(self, model: str) -> None:
        self._model = model

    def set_query_endpoint(self, url: str | None, token: str | None) -> None:
        self._set_query_endpoint(url, token)

    def _set_query_endpoint(self, url: str | None, token: str | None) -> None:
        base = url.rstrip("/") if url else self._url.rsplit("/chat/completions", 1)[0]
        self._query_url = f"{base}/chat/completions"
        q_token = token if token is not None else self._token
        self._query_headers = {"Authorization": f"Bearer {q_token}"} if q_token else {}

    def analyze(
        self,
        frames_by_camera: dict[str, list[tuple[datetime, np.ndarray]]],
        boxes_by_camera: dict[str, list[tuple[int, int, int, int]]],
    ) -> tuple[str, list[np.ndarray]]:
        prompt = _ANALYZE_PROMPT_PATH.read_text().format(dog_description=self._dog_description)
        content: list[dict] = [{"type": "text", "text": prompt}]
        sampled_frames: list[np.ndarray] = []

        for camera, frames in frames_by_camera.items():
            boxes = boxes_by_camera.get(camera, [])
            for ts, frame in _sample_tiered(frames, self._frame_sampling):
                cropped = _crop(frame, boxes, self._crop_padding) if boxes else frame
                sampled_frames.append(cropped)
                content.append({"type": "text", "text": f"{camera} @ {ts.strftime('%H:%M:%S.%f')[:-3]}"})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_encode(cropped)}"},
                })

        response = requests.post(
            self._url,
            headers=self._headers,
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": self._max_tokens,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"], sampled_frames

    def summarize(self, prompt: str = "", max_tokens: int = 200, model: str | None = None, query: bool = False, messages: list[dict] | None = None) -> str:
        url = self._query_url if query else self._url
        headers = self._query_headers if query else self._headers
        response = requests.post(
            url,
            headers=headers,
            json={
                "model": model or self._model,
                "messages": messages if messages is not None else [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""

    def detect_dog(self, frames_by_camera: dict[str, np.ndarray]) -> list[str]:
        content: list[dict] = [{"type": "text", "text": _DETECT_PROMPT_PATH.read_text().format(dog_description=self._dog_description)}]
        for camera, frame in frames_by_camera.items():
            content.append({"type": "text", "text": camera})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{_encode(frame)}"},
            })
        response = requests.post(
            self._url,
            headers=self._headers,
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": self._max_tokens,
            },
            timeout=30,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        return json.loads(extract_json(text)).get("cameras_with_dog", [])


def extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def _crop(frame: np.ndarray, boxes: list[tuple[int, int, int, int]], padding: float) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return frame[y1:y2, x1:x2]


def _sample(
    frames: list[tuple[datetime, np.ndarray]], n: int
) -> list[tuple[datetime, np.ndarray]]:
    if not frames or n <= 0 or len(frames) <= n:
        return frames
    if n == 1:
        return [frames[-1]]
    indices = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    return [frames[i] for i in indices]


def _sample_tiered(
    frames: list[tuple[datetime, np.ndarray]],
    tiers: list[tuple[float, float]],
) -> list[tuple[datetime, np.ndarray]]:
    if not frames:
        return []
    latest_ts = frames[-1][0]
    result: list[tuple[datetime, np.ndarray]] = []
    boundary = latest_ts
    for seconds, fps in tiers:
        start = boundary - timedelta(seconds=seconds)
        bucket = [(ts, f) for ts, f in frames if start <= ts < boundary]
        n = round(seconds * fps)
        if n > 0 and bucket:
            result = _sample(bucket, n) + result
        boundary = start
    return result


def _encode(frame: np.ndarray) -> str:
    h, w = frame.shape[:2]
    if w > _LLM_MAX_WIDTH or h > _LLM_MAX_HEIGHT:
        scale = min(_LLM_MAX_WIDTH / w, _LLM_MAX_HEIGHT / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    return base64.b64encode(buf).decode()
