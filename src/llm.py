import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests

import logging

from config import LLMEndpointConfig
from utils import encode_frame

logger = logging.getLogger(__name__)


class LLMResponseError(RuntimeError):
    """
    An LLM endpoint returned a well-formed HTTP response that carries no
    completion. OpenRouter answers HTTP 200 with a body of
    `{"error": {...}, "user_id": ...}` when an upstream provider fails after the
    request was accepted, so a non-2xx status is not enough to detect failure.

    `response` mirrors `requests.HTTPError.response` so callers can pull the
    status and provider message off the exception the same way.
    """

    def __init__(self, message: str, response: requests.Response):
        super().__init__(message)
        self.response = response


# Max seconds apart two cameras' frames may be to share a single "t=" instant
# header. Purely a labeling/grouping window — frames outside it are still sent,
# just under their own header. Loosen it to group more cameras per instant.
_FRAME_SYNC_TOLERANCE = 0.1

# Hard wall-clock cap on the verification pass. It only trims false positives,
# and a caller that falls back to the pass 1 result is better than an alert that
# arrives late, so this is deliberately far tighter than the primary timeout.
_VERIFY_TIMEOUT = 5

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_ANALYZE_PROMPT_PATH = _PROMPTS_DIR / "analyze_prompt.txt"
_DETECT_PROMPT_PATH = _PROMPTS_DIR / "detect_prompt.txt"

# Seeded into the think block (via continue_final_message) to prime terse,
# on-task Chinese reasoning. `<think>\n` is a control prefix stripped from display.
_REASONING_SEED_VISIBLE = "位置:"
_REASONING_SEED_OPEN = "<think>\n" + _REASONING_SEED_VISIBLE

# Printable-ASCII pattern forces English output at the grammar level, so Chinese
# reasoning can't bleed into the answer fields.
_ASCII = "^[ -~]+$"
_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "pattern": _ASCII},
        "summary": {"type": "string", "pattern": _ASCII},
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
    _ANALYZE_PARSE_ATTEMPTS = 2

    def __init__(
        self,
        config: LLMEndpointConfig,
        dog_description: str,
        verify_model: str | None = None,
        verify_url: str | None = None,
        verify_token: str | None = None,
    ):
        self._vision_model = config.vision_model
        self._fast_model = config.fast_model
        self._memory_model = config.memory_model
        self._dog_description = dog_description
        self._frame_sampling = [(t["seconds"], t["fps"]) for t in config.frame_sampling]
        self._crop_padding = config.crop_padding
        self._max_frame_width = config.max_frame_width
        self._max_frame_height = config.max_frame_height
        self._max_tokens = config.max_tokens
        self._temperature = config.temperature
        self._reasoning_budget = config.reasoning_budget
        self._vision_url, self._vision_headers = self._endpoint(config.vision_url, config.vision_token)
        self._fast_url, self._fast_headers = self._endpoint(config.fast_url, config.fast_token)
        self._memory_url, self._memory_headers = self._endpoint(config.memory_url, config.memory_token)
        # Verification (second-pass) model. Falls back to the vision model/endpoint
        # when not configured, so an unset verify_* config reproduces the old
        # behavior of re-running the same vision model.
        self._verify_model = verify_model or config.vision_model
        self._verify_url, self._verify_headers = self._endpoint(
            verify_url or config.vision_url,
            verify_token if verify_url else config.vision_token,
        )
        # Pool used purely to enforce a hard wall-clock cap on each request: the
        # requests `timeout` is per-socket-read, so a server that accepts the
        # connection then trickles (or never finishes) bytes can hang far longer
        # than `timeout`. We submit the request here and bound it with
        # future.result(deadline) so the caller always recovers.
        self._post_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm-post")

    @staticmethod
    def _endpoint(url: str, token: str | None) -> tuple[str, dict]:
        return (
            f"{url.rstrip('/')}/chat/completions", 
            {"Authorization": f"Bearer {token}"} if token else {}
        )

    def _post(
        self, url: str, headers: dict, payload: dict, timeout: int = 30,
        deadline: float | None = None, stream: bool = False,
    ):
        # `timeout` is the per-socket connect/read timeout passed to requests;
        # `deadline` is an absolute wall-clock cap that catches a server which
        # trickles bytes slower than the read timeout but never stalls long
        # enough to trip it. By default give the pool worker a little headroom
        # over the socket timeout so requests' own timeout fires first when it
        # can; callers needing a hard cap can pass `deadline` explicitly.
        #
        # `stream=True` returns `(content, usage, timing)` instead of
        # `(content, usage)`: the streamed path also measures time-to-first-token
        # so callers can separate prefill from generation time. Either way the
        # whole request runs in the pool worker, so `deadline` still bounds it.
        if deadline is None:
            deadline = timeout + 15
        worker = self._do_post_stream if stream else self._do_post
        future = self._post_pool.submit(worker, url, headers, payload, timeout)
        try:
            return future.result(timeout=deadline)
        except FuturesTimeout:
            future.cancel()
            raise TimeoutError(f"LLM request to {url} exceeded {deadline}s wall-clock deadline")

    @staticmethod
    def _do_post_stream(url: str, headers: dict, payload: dict, timeout: int) -> tuple[str, dict, dict]:
        """
        Stream the completion so prefill and generation time can be separated.

        vLLM (and OpenRouter) report token *counts* in `usage` but never
        per-phase timing, so the only way to get a generation tok/s that isn't
        diluted by the prompt-eval phase — which dominates for image prompts —
        is to time it here. Time-to-first-token approximates the prefill cost;
        the span from the first to the last content token is the true generation
        time. `stream_options.include_usage` asks for a final usage-only chunk;
        a server that omits it just leaves `usage` empty and callers degrade to
        showing no token stats rather than wrong ones.

        Returns `(content, usage, timing)` where `timing` has `ttft` and
        `generation_time` in seconds (either may be None if no tokens streamed).
        """
        body = {**payload, "stream": True, "stream_options": {"include_usage": True}}
        start = time.monotonic()
        first_token_at: float | None = None
        last_token_at: float | None = None
        parts: list[str] = []
        reasoning_parts: list[str] = []
        reasoning_token_count = 0
        usage: dict = {}
        with requests.post(
            url, headers=headers, json=body, stream=True,
            timeout=(min(10, timeout), timeout),
        ) as response:
            response.raise_for_status()
            for raw in response.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                error = chunk.get("error")
                if error:
                    message = error.get("message") if isinstance(error, dict) else None
                    raise LLMResponseError(message or str(error), response)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    # A reasoning parser (vLLM --reasoning-parser) streams the
                    # thinking trace separately from the final answer `content`.
                    # The field name varies: this vLLM build and OpenRouter use
                    # `reasoning`; older vLLM used `reasoning_content`. Advance the
                    # token clock on either kind so generation_time covers the full
                    # decode (reasoning + answer) rather than the answer alone —
                    # otherwise the reasoning time lands in ttft/prefill and the
                    # phases stop reconciling with the total.
                    reasoning_piece = delta.get("reasoning") or delta.get("reasoning_content")
                    piece = delta.get("content")
                    if reasoning_piece or piece:
                        now = time.monotonic()
                        if first_token_at is None:
                            first_token_at = now
                        last_token_at = now
                    if reasoning_piece:
                        reasoning_parts.append(reasoning_piece)
                        # Streamed one delta per token; count them so the UI can
                        # show reasoning's share when `usage` omits the breakdown.
                        reasoning_token_count += 1
                    if piece:
                        parts.append(piece)
        timing = {
            "ttft": (first_token_at - start) if first_token_at is not None else None,
            "generation_time": (
                last_token_at - first_token_at
                if first_token_at is not None and last_token_at is not None
                else None
            ),
            "reasoning": "".join(reasoning_parts) or None,
            "reasoning_tokens": reasoning_token_count or None,
        }
        return "".join(parts), usage, timing

    @staticmethod
    def _do_post(url: str, headers: dict, payload: dict, timeout: int) -> tuple[str, dict]:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            # Never let the connect phase alone outlast the caller's budget.
            timeout=(min(10, timeout), timeout),
        )
        response.raise_for_status()
        try:
            body = response.json()
        except ValueError as e:
            raise LLMResponseError(
                f"non-JSON response: {response.text[:200]!r}", response
            ) from e

        error = body.get("error")
        if error:
            message = error.get("message") or str(error)
            provider = (error.get("metadata") or {}).get("provider_name")
            code = error.get("code")
            detail = ", ".join(
                str(part) for part in (message, provider and f"provider={provider}", code and f"code={code}") if part
            )
            raise LLMResponseError(detail, response)

        choices = body.get("choices")
        if not choices:
            raise LLMResponseError(f"no choices in response: {body!r}"[:300], response)
        # `usage` carries token counts (prompt/completion/total) on OpenAI-compatible
        # endpoints; it's optional, so callers must tolerate an empty dict.
        return choices[0]["message"]["content"] or "", body.get("usage") or {}

    @staticmethod
    def _no_reasoning(url: str) -> dict:
        """
        Request-body fragment that suppresses reasoning tokens on `url`.

        The switch is endpoint-specific: OpenRouter reads a nested `reasoning`
        object, while vLLM toggles Qwen thinking through the chat template.
        Both silently drop unknown top-level keys, so sending the wrong one
        fails open into a reasoning model rather than erroring.
        """
        if "openrouter.ai" in url:
            return {"reasoning": {"effort": "none", "exclude": True}}
        return {"chat_template_kwargs": {"enable_thinking": False}}

    def _reasoning_params(self, url: str) -> dict:
        """
        Request-body fragment enabling bounded reasoning on `url`.

        A budget of 0 (or less) disables reasoning entirely, delegating to
        `_no_reasoning`. Otherwise a hard thinking-token cap is applied: vLLM
        enforces it natively via the top-level `thinking_token_budget` sampling
        param (which requires the server's `--reasoning-parser`, forcing `</think>`
        at the budget), while OpenRouter caps reasoning via nested
        `reasoning.max_tokens`. Endpoints silently drop the key that doesn't
        apply, so the wrong one fails open rather than erroring.
        """
        if self._reasoning_budget <= 0:
            return self._no_reasoning(url)
        if "openrouter.ai" in url:
            return {"reasoning": {"max_tokens": self._reasoning_budget}}
        return {
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking_token_budget": self._reasoning_budget,
        }

    @staticmethod
    def _seeds_thinking(reasoning: bool, url: str) -> bool:
        # continue_final_message priming is a vLLM feature; skip it on OpenRouter.
        return reasoning and "openrouter.ai" not in url

    def _json_schema_payload(
        self, messages: list[dict], name: str, schema: dict, url: str,
        model: str | None = None, reasoning: bool = False,
    ) -> dict:
        seed: dict = {}
        if self._seeds_thinking(reasoning, url):
            messages = messages + [{"role": "assistant", "content": _REASONING_SEED_OPEN}]
            seed = {"continue_final_message": True, "add_generation_prompt": False}
        return {
            "model": model or self._vision_model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            **(self._reasoning_params(url) if reasoning else self._no_reasoning(url)),
            **seed,
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
        frames_by_camera: dict[str, list[tuple[datetime, np.ndarray]]],
        boxes_by_camera: dict[str, list[tuple[int, int, int, int]]],
        verify: bool = False,
    ) -> tuple[str, list[np.ndarray], dict[str, list[np.ndarray]], list[dict], dict]:
        """
        Crop and run LLM analysis on video frames.

        Args:
            frames_by_camera: dict mapping camera name to list of
                (timestamp, frame) tuples
            boxes_by_camera: dict mapping camera name to list of bounding boxes
                (x1, y1, x2, y2)
            verify: when True, use the verification model/endpoint instead of the
                primary vision model (used for the second confirmation pass).

        Returns:
            str: LLM response content JSON as string
            list[np.ndarray]: list of sampled frames (camera-major)
            dict[str, list[np.ndarray]]: sampled frames grouped by camera
            list[dict]: list of messages sent to LLM
            dict: inference stats — `prompt_tokens`, `completion_tokens`,
                `total_tokens` (token counts, or None when the endpoint omits
                `usage`) plus `prefill_time` and `generation_time` in seconds
                (or None when nothing streamed). Generation time is measured via
                streaming so it excludes the prompt-eval/prefill phase. Also
                `reasoning`: the model's thinking trace when reasoning is on
                (None otherwise).
        """
        prompt = _ANALYZE_PROMPT_PATH.read_text().format(
            dog_description=self._dog_description
        )
        content, sampled_frames, sampled_by_camera = _build_frame_content(
            frames_by_camera, boxes_by_camera,
            self._frame_sampling, self._crop_padding,
            self._max_frame_width, self._max_frame_height,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]
        model = self._verify_model if verify else self._vision_model
        url = self._verify_url if verify else self._vision_url
        headers = self._verify_headers if verify else self._vision_headers
        payload = self._json_schema_payload(
            messages,
            name="dog_analysis",
            schema=_ANALYSIS_SCHEMA,
            url=url,
            model=model,
            reasoning=self._reasoning_budget > 0,
        )

        # Retry the main pass when the model returns no JSON (a thinking model
        # rarely emits only reasoning); verify is single-shot since its failure
        # just falls back to pass 1.
        timeout = _VERIFY_TIMEOUT if verify else 60
        deadline = _VERIFY_TIMEOUT if verify else None
        attempts = 1 if verify else self._ANALYZE_PARSE_ATTEMPTS
        for attempt in range(attempts):
            content, usage, timing = self._post(
                url, headers, payload, timeout=timeout, deadline=deadline, stream=True,
            )
            if extract_json(content).lstrip().startswith("{"):
                break
            if attempt + 1 < attempts:
                logger.warning("Analyze returned no JSON, retrying (%d/%d)", attempt + 1, attempts)
        reasoning = timing.get("reasoning")
        if reasoning and self._seeds_thinking(self._reasoning_budget > 0, url):
            reasoning = _REASONING_SEED_VISIBLE + reasoning
        stats = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "prefill_time": timing.get("ttft"),
            "generation_time": timing.get("generation_time"),
            "reasoning": reasoning,
            # Reasoning is part of the decode phase, so `completion_tokens`
            # already includes it; this is just the reasoning share. Prefer the
            # endpoint's own breakdown, falling back to the count of streamed
            # reasoning deltas when `usage` omits it (as this vLLM build does).
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ) or timing.get("reasoning_tokens"),
        }
        return content, sampled_frames, sampled_by_camera, messages, stats

    def summarize(
        self,
        prompt: str = "",
        max_tokens: int = 8192,
        model: str | None = None,
        endpoint: str = "fast",
        messages: list[dict] | None = None,
    ) -> str:
        """
        Summarize a minute's
        """
        url, headers = {
            "fast": (self._fast_url, self._fast_headers),
            "memory": (self._memory_url, self._memory_headers),
        }[endpoint]
        content, _ = self._post(
            url,
            headers,
            {
                "model": model or self._vision_model,
                "messages": messages if messages is not None else [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                **self._no_reasoning(url),
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
                "image_url": {"url": f"data:image/jpeg;base64,{encode_frame(frame, self._max_frame_width, self._max_frame_height)}"},
            })
        payload = self._json_schema_payload(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            name="dog_detection",
            schema=_DETECTION_SCHEMA,
            url=self._vision_url,
        )

        def _call() -> set[str]:
            for _ in range(self._DETECT_PARSE_ATTEMPTS):
                content, _ = self._post(self._vision_url, self._vision_headers, payload)
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

    def set_verify_model(self, model: str) -> None:
        self._verify_model = model

    def set_fast_model(self, model: str) -> None:
        self._fast_model = model

    def set_memory_model(self, model: str) -> None:
        self._memory_model = model

    def set_vision_endpoint(self, url: str, token: str | None) -> None:
        self._vision_url, self._vision_headers = self._endpoint(url, token)

    def set_verify_endpoint(self, url: str, token: str | None) -> None:
        self._verify_url, self._verify_headers = self._endpoint(url, token)

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

    def set_max_frame_size(self, width: int, height: int) -> None:
        self._max_frame_width = width
        self._max_frame_height = height

    def set_max_tokens(self, max_tokens: int) -> None:
        self._max_tokens = max_tokens

    def set_temperature(self, temperature: float) -> None:
        self._temperature = temperature

    def set_reasoning_budget(self, reasoning_budget: int) -> None:
        self._reasoning_budget = reasoning_budget


############  Helper functions below  ############

def extract_json(text: str) -> str:
    """
    Extract JSON (as string) from LLM response content, which may contain 
    reasoning or other text (although it shouldn't if LLM follows the
    instructions correctly).
    """
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
    frames_by_camera: dict[str, list[tuple[datetime, np.ndarray]]],
    boxes_by_camera: dict[str, list[tuple[int, int, int, int]]],
    frame_sampling: list[tuple[float, float]],
    crop_padding: float,
    max_frame_width: int,
    max_frame_height: int,
) -> tuple[list[dict], list[np.ndarray], dict[str, list[np.ndarray]]]:
    """
    Build LLM message content and the matching sampled frames from camera frames.

    Each camera is sampled independently per the tiers, then all sampled frames
    are pooled and emitted time-major (ascending timestamp). Frames that fall
    within `_FRAME_SYNC_TOLERANCE` of each other are grouped under a single
    "t=<offset>s" header as one instant seen from overlapping cameras; a frame
    that lines up with nothing simply gets its own header. Every sampled frame
    is sent to the LLM — grouping only affects labeling, never inclusion.
    Timestamps are normalized so the earliest sampled frame is 0, and each frame
    is cropped to the detection boxes (with padding) when boxes are present.

    The returned frame list is ordered camera-major (each camera's frames in
    time order, one camera after another) rather than in the time-major content
    order, so the compiled alert/eval video stays watchable as a coherent clip
    per camera instead of flickering between angles. The per-camera grouping is
    also returned so callers can compile one clip per source.
    """
    pooled: list[tuple[datetime, str, np.ndarray]] = []
    for camera, frames in frames_by_camera.items():
        boxes = boxes_by_camera.get(camera, [])
        for ts, frame in _sample_tiered(frames, frame_sampling):
            display = _crop(frame, boxes, crop_padding) if boxes else frame
            pooled.append((ts, camera, display))

    content: list[dict] = []
    if not pooled:
        return content, [], {}
    pooled.sort(key=lambda item: item[0])
    base_ts = pooled[0][0]

    frames_by_camera_out: dict[str, list[np.ndarray]] = {}
    group: list[tuple[str, np.ndarray]] = []
    group_start: datetime | None = None
    group_cameras: set[str] = set()

    def flush() -> None:
        if not group:
            return
        offset = (group_start - base_ts).total_seconds()
        content.append({"type": "text", "text": f"t={offset:.2f}s"})
        for camera, display in group:
            frames_by_camera_out.setdefault(camera, []).append(display)
            content.append({"type": "text", "text": camera})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_frame(display, max_frame_width, max_frame_height)}"},
            })

    for ts, camera, display in pooled:
        in_window = (
            group_start is not None
            and (ts - group_start).total_seconds() <= _FRAME_SYNC_TOLERANCE
        )
        if in_window and camera not in group_cameras:
            group.append((camera, display))
            group_cameras.add(camera)
        else:
            flush()
            group = [(camera, display)]
            group_start = ts
            group_cameras = {camera}
    flush()

    sampled_frames = [frame for frames in frames_by_camera_out.values() for frame in frames]
    return content, sampled_frames, frames_by_camera_out


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
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]

def _sample(
    frames: list[tuple[datetime, np.ndarray]],
    num_frames: int,
) -> list[tuple[datetime, np.ndarray]]:
    """
    Sample `num_frames` evenly spaced frames from the list, including the first
    and last.
    """
    if not frames or num_frames <= 0 or len(frames) <= num_frames:
        return frames
    if num_frames == 1:
        return [frames[-1]]
    indices = [round(i * (len(frames) - 1) / (num_frames - 1)) for i in range(num_frames)]
    return [frames[i] for i in indices]

def _sample_tiered(
    frames: list[tuple[datetime, np.ndarray]],
    tiers: list[tuple[float, float]],
) -> list[tuple[datetime, np.ndarray]]:
    """
    Sample frames from the list according to the specified tiers.

    Args:
        frames (list[tuple[datetime, np.ndarray]]): List of (timestamp,
            frame) tuples sorted by timestamp ascending
        tiers (list[tuple[float, float]]): List of (seconds, fps) tuples
            specifying the sampling tiers in order from latest to oldest.
            For example, [(3, 5), (7, 1)] samples 3 most recent seconds at
            5 fps, then the 7 seconds before that at 1 fps. Total seconds
            must be greater than the age of the oldest frame.
    Returns:
        list[tuple[datetime, np.ndarray]]: List of (timestamp, frame) tuples
            for the sampled frames.
    """
    if not frames:
        return []
    latest_ts = frames[-1][0]
    result: list[tuple[datetime, np.ndarray]] = []
    boundary = latest_ts
    for seconds, fps in tiers:
        start = boundary - timedelta(seconds=seconds)
        bucket = [item for item in frames if start <= item[0] < boundary]
        n = round(seconds * fps)
        if n > 0 and bucket:
            result = _sample(bucket, n) + result
        boundary = start
    return result
