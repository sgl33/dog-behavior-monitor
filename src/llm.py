import json
import re
from collections.abc import Callable
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

_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "summary": {"type": "string"},
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
    },
    "required": ["description", "summary", "score"],
    "additionalProperties": False,
}

_DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "cameras_with_dog": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["cameras_with_dog"],
    "additionalProperties": False,
}


class LLMClient:
    _DETECT_PARSE_ATTEMPTS = 3

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
        return (
            f"{url.rstrip('/')}/chat/completions", 
            ({"Authorization": f"Bearer {token}"} if token else {})
        )

    def _post(self, url: str, headers: dict, payload: dict, timeout: int = 30) -> str:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""

    def _json_schema_payload(self, messages: list[dict], name: str, schema: dict) -> dict:
        return {
            "model": self._vision_model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "enable_thinking": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }

    def analyze(
        self,
        frames_by_camera: dict[str, list[tuple[datetime, np.ndarray, str]]],
        boxes_by_camera: dict[str, list[tuple[int, int, int, int]]],
    ) -> tuple[str, list[np.ndarray], list[dict]]:
        """
        Crop and run LLM analysis on video frames.

        Args:
            frames_by_camera: dict mapping camera name to list of 
                (timestamp, frame, encoded frame) tuples
            boxes_by_camera: dict mapping camera name to list of bounding boxes 
                (x1, y1, x2, y2)

        Returns: 
            str: LLM response content JSON as string
            list[np.ndarray]: list of sampled frames
            list[dict]: list of messages sent to LLM
        """
        prompt = _ANALYZE_PROMPT_PATH.read_text().format(
            dog_description=self._dog_description
        )
        content, sampled_frames = _build_frame_content(
            frames_by_camera, boxes_by_camera, 
            self._frame_sampling, self._crop_padding
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]
        payload = self._json_schema_payload(
            messages,
            name="dog_analysis",
            schema=_ANALYSIS_SCHEMA,
        )

        content = self._post(self._vision_url, self._vision_headers, payload, timeout=60)
        return content, sampled_frames, messages

    def summarize(
        self,
        prompt: str = "",
        max_tokens: int = 1024,
        model: str | None = None,
        endpoint: str = "fast",
        messages: list[dict] | None = None,
    ) -> str:
        content = self._post(
            self._fast_url,
            self._fast_headers,
            {
                "model": model or self._vision_model,
                "messages": messages if messages is not None else [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "enable_thinking": False,
            },
        )
        logger.info("LLM %s: %s", endpoint, content)
        return content

    def detect_dog(
        self,
        frames_by_camera: dict[str, np.ndarray],
        should_abort: Callable[[], bool] | None = None,
    ) -> list[str]:
        prompt = _DETECT_PROMPT_PATH.read_text().format(dog_description=self._dog_description)
        content: list[dict] = []
        for camera, frame in frames_by_camera.items():
            content.append({"type": "text", "text": camera})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_frame(frame)}"},
            })
        payload = self._json_schema_payload(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            name="dog_detection",
            schema=_DETECTION_SCHEMA,
        )

        def _call() -> set[str]:
            for _ in range(self._DETECT_PARSE_ATTEMPTS):
                content = self._post(self._vision_url, self._vision_headers, payload)
                logger.info("LLM detect: %s", content)
                try:
                    return set(json.loads(extract_json(content)).get("cameras_with_dog", []))
                except (json.JSONDecodeError, ValueError):
                    continue
            return set()

        first = _call()
        if not first:
            return []
        if should_abort is not None and should_abort():
            logger.info("YOLO detected during fallback, aborting LLM confirmation")
            return []
        second = _call()
        return sorted(first & second)

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


############  Helper functions below  ############

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


def _build_frame_content(
    frames_by_camera: dict[str, list[tuple[datetime, np.ndarray, str]]],
    boxes_by_camera: dict[str, list[tuple[int, int, int, int]]],
    frame_sampling: list[tuple[float, float]],
    crop_padding: float,
) -> tuple[list[dict], list[np.ndarray]]:
    """
    Build LLM message content and the matching sampled frames from camera frames.

    Frames are sampled per tier, cropped to the detection boxes (with padding)
    when boxes are present, and emitted as interleaved timestamp/image entries.
    """
    content: list[dict] = []
    sampled_frames: list[np.ndarray] = []

    for camera, frames in frames_by_camera.items():
        boxes = boxes_by_camera.get(camera, [])
        for ts, frame, encoded in _sample_tiered(frames, frame_sampling):
            if boxes:
                display = _crop(frame, boxes, crop_padding)
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

    return content, sampled_frames


def _crop(
    frame: np.ndarray, 
    boxes: list[tuple[int, int, int, int]], 
    padding: float
) -> np.ndarray:
    """Return a cropped region of `frame` plus padding."""
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
