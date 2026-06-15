import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

import logging

from config import LLMEndpointConfig
from utils import encode_frame

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_ANALYZE_PROMPT_PATH = _PROMPTS_DIR / "analyze_prompt.txt"
_DETECT_PROMPT_PATH = _PROMPTS_DIR / "detect_prompt.txt"


class LLMClient:
    def __init__(self, config: LLMEndpointConfig, dog_description: str):
        self._vision_model = config.vision_model
        self._fast_model = config.fast_model
        self._memory_model = config.memory_model
        self._dog_description = dog_description
        self._frame_sampling = [(t["seconds"], t["fps"]) for t in config.frame_sampling]
        self._crop_padding = config.crop_padding
        self._max_tokens = config.max_tokens
        self._vision_url, self._vision_headers = self._endpoint(config.vision_url, config.vision_token)
        self._fast_url, self._fast_headers = self._endpoint(config.fast_url, config.fast_token)
        self._memory_url, self._memory_headers = self._endpoint(config.memory_url, config.memory_token)

    @staticmethod
    def _endpoint(url: str, token: str | None) -> tuple[str, dict]:
        return f"{url.rstrip('/')}/chat/completions", ({"Authorization": f"Bearer {token}"} if token else {})

    @property
    def fast_model(self) -> str:
        return self._fast_model

    @property
    def memory_model(self) -> str:
        return self._memory_model

    def set_vision_model(self, model: str) -> None:
        self._vision_model = model

    def set_fast_model(self, model: str) -> None:
        self._fast_model = model

    def set_memory_model(self, model: str) -> None:
        self._memory_model = model

    def set_vision_endpoint(self, url: str, token: str | None) -> None:
        self._vision_url, self._vision_headers = self._endpoint(url, token)

    def set_fast_endpoint(self, url: str, token: str | None) -> None:
        self._fast_url, self._fast_headers = self._endpoint(url, token)

    def set_memory_endpoint(self, url: str, token: str | None) -> None:
        self._memory_url, self._memory_headers = self._endpoint(url, token)

    def set_dog_description(self, description: str) -> None:
        self._dog_description = description

    def set_frame_sampling(self, tiers: list[dict]) -> None:
        self._frame_sampling = [(t["seconds"], t["fps"]) for t in tiers]

    def set_crop_padding(self, padding: float) -> None:
        self._crop_padding = padding

    def set_max_tokens(self, max_tokens: int) -> None:
        self._max_tokens = max_tokens

    def analyze(
        self,
        frames_by_camera: dict[str, list[tuple[datetime, np.ndarray, str]]],
        boxes_by_camera: dict[str, list[tuple[int, int, int, int]]],
    ) -> tuple[str, list[np.ndarray], list[dict]]:
        prompt = _ANALYZE_PROMPT_PATH.read_text().format(dog_description=self._dog_description)
        content: list[dict] = []
        sampled_frames: list[np.ndarray] = []

        for camera, frames in frames_by_camera.items():
            boxes = boxes_by_camera.get(camera, [])
            for ts, frame, encoded in _sample_tiered(frames, self._frame_sampling):
                if boxes:
                    display = _crop(frame, boxes, self._crop_padding)
                    img_b64 = encode_frame(display)
                else:
                    display = frame
                    img_b64 = encoded
                sampled_frames.append(display)
                content.append({"type": "text", "text": f"{camera} @ {ts.strftime('%H:%M:%S.%f')[:-3]}"})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                })

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]
        payload = {
            "model": self._vision_model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "enable_thinking": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "dog_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "summary": {"type": "string"},
                            "score": {"type": "integer", "minimum": 0, "maximum": 10},
                        },
                        "required": ["description", "summary", "score"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        try:
            response = requests.post(
                self._vision_url,
                headers=self._vision_headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
        except Exception:
            time.sleep(2)
            raise
        content = response.json()["choices"][0]["message"]["content"]
        # logger.info("LLM analyze: %s", content)
        return content, sampled_frames, messages

    def summarize(
        self,
        prompt: str = "",
        max_tokens: int = 1024,
        model: str | None = None,
        endpoint: str = "fast",
        messages: list[dict] | None = None,
    ) -> str:
        _endpoints = {
            "vision": (self._vision_url, self._vision_headers),
            "fast":   (self._fast_url,   self._fast_headers),
            "memory": (self._memory_url, self._memory_headers),
        }
        url, headers = _endpoints.get(endpoint, _endpoints["fast"])
        response = requests.post(
            url,
            headers=headers,
            json={
                "model": model or self._vision_model,
                "messages": messages if messages is not None else [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "enable_thinking": False,
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"] or ""
        logger.info("LLM %s: %s", endpoint, content)
        return content

    def detect_dog(self, frames_by_camera: dict[str, np.ndarray]) -> list[str]:
        prompt = _DETECT_PROMPT_PATH.read_text().format(dog_description=self._dog_description)
        content: list[dict] = []
        for camera, frame in frames_by_camera.items():
            content.append({"type": "text", "text": camera})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_frame(frame)}"},
            })
        payload = {
            "model": self._vision_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": self._max_tokens,
            "enable_thinking": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "dog_detection",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "cameras_with_dog": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["cameras_with_dog"],
                        "additionalProperties": False,
                    },
                },
            },
        }

        def _call() -> set[str]:
            for _ in range(3):
                r = requests.post(self._vision_url, headers=self._vision_headers, json=payload, timeout=30)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"] or ""
                logger.info("LLM detect: %s", content)
                try:
                    return set(json.loads(extract_json(content)).get("cameras_with_dog", []))
                except (json.JSONDecodeError, ValueError):
                    continue
            return set()

        first = _call()
        if not first:
            return []
        second = _call()
        return sorted(first & second)


def extract_json(text: str) -> str:
    # Strip reasoning blocks emitted by thinking models (e.g. Gemma, QwQ)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
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
    frames: list[tuple[datetime, np.ndarray, str]],
    tiers: list[tuple[float, float]],
) -> list[tuple[datetime, np.ndarray, str]]:
    if not frames:
        return []
    latest_ts = frames[-1][0]
    result: list[tuple[datetime, np.ndarray, str]] = []
    boundary = latest_ts
    for seconds, fps in tiers:
        start = boundary - timedelta(seconds=seconds)
        bucket = [item for item in frames if start <= item[0] < boundary]
        n = round(seconds * fps)
        if n > 0 and bucket:
            result = _sample(bucket, n) + result
        boundary = start
    return result
