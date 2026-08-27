#!/usr/bin/env python3
"""Streaming VLM scene descriptions for live GUI display."""

from __future__ import annotations

import base64
import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SCENE_DESCRIPTION_SYSTEM_PROMPT = (
    "You are an intelligent scene analysis assistant for an industrial robotics platform."
)
SCENE_DESCRIPTION_PROMPT = """You are an intelligent scene analysis assistant for an industrial robotics platform.
Analyze the provided image and provide a concise, informative description of the scene.

Focus on:
- What objects are visible and their approximate locations
- Any people present and what they appear to be doing
- The overall environment/setting (warehouse, office, outdoor, etc.)
- Any notable activities, hazards, or points of interest
- Spatial relationships between key objects

Keep your description to 2-4 sentences. Be specific and factual. Do not speculate about things you cannot clearly see."""


@dataclass
class SceneDescription:
    """A natural language scene description from the VLM."""

    camera_id: int
    description: str
    confidence: float
    timestamp: float
    frame_token: Tuple[str, int, int]
    latency_ms: float
    is_final: bool = True
    error: Optional[str] = None


class VLMDetectorError(RuntimeError):
    """Raised when the VLM request or response fails."""


class OpenAISceneDescriber:
    """Query a vLLM OpenAI-compatible endpoint for scene descriptions."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float,
        max_tokens: int,
        jpeg_quality: int,
        image_max_side_px: int,
    ) -> None:
        if not base_url:
            raise ValueError("VLM base URL is empty")
        if not model:
            raise ValueError("VLM model name is empty")

        self._base_url = base_url
        self._api_key = api_key or "EMPTY"
        self._model = model
        self._timeout_s = float(timeout_s)
        self._max_tokens = max(1, int(max_tokens))
        self._jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self._image_max_side_px = max(0, int(image_max_side_px))
        self._client = None

    def describe_scene(
        self,
        color_image: np.ndarray,
        *,
        camera_id: int,
        frame_token: Tuple[str, int, int],
    ) -> SceneDescription:
        """Return a complete scene description for one RGB image."""
        final_description: Optional[SceneDescription] = None
        for description in self.stream_scene(
            color_image,
            camera_id=camera_id,
            frame_token=frame_token,
        ):
            final_description = description

        if final_description is None or not final_description.description.strip():
            raise VLMDetectorError("VLM response did not include text content")
        return final_description

    def stream_scene(
        self,
        color_image: np.ndarray,
        *,
        camera_id: int,
        frame_token: Tuple[str, int, int],
    ) -> Iterator[SceneDescription]:
        """Yield partial scene descriptions as streamed VLM tokens arrive."""
        if (
            not isinstance(color_image, np.ndarray)
            or color_image.ndim != 3
            or color_image.shape[2] != 3
            or color_image.size == 0
        ):
            raise ValueError("Scene describer color image must be a non-empty HxWx3 array")

        jpeg_data_url = self._encode_rgb_data_url(color_image)
        response_started_s = time.monotonic()
        frame_timestamp_s = time.time()
        stream = None
        try:
            stream = self._ensure_client().chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": SCENE_DESCRIPTION_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": SCENE_DESCRIPTION_PROMPT},
                            {"type": "image_url", "image_url": {"url": jpeg_data_url}},
                        ],
                    },
                ],
                temperature=0.2,
                max_tokens=self._max_tokens,
                stream=True,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        except Exception as exc:
            raise VLMDetectorError(f"VLM scene description request failed: {exc}") from exc

        accumulated = ""
        try:
            for chunk in stream:
                delta = self._extract_stream_delta(chunk)
                if not delta:
                    continue
                accumulated += delta
                visible_text = accumulated.strip()
                if not visible_text:
                    continue
                yield SceneDescription(
                    camera_id=camera_id,
                    description=visible_text,
                    confidence=1.0,
                    timestamp=frame_timestamp_s,
                    frame_token=frame_token,
                    latency_ms=(time.monotonic() - response_started_s) * 1000.0,
                    is_final=False,
                )
        except Exception as exc:
            raise VLMDetectorError(f"VLM scene description stream failed: {exc}") from exc

        final_text = accumulated.strip()
        if not final_text:
            raise VLMDetectorError("VLM response did not include text content")

        yield SceneDescription(
            camera_id=camera_id,
            description=final_text,
            confidence=1.0,
            timestamp=time.time(),
            frame_token=frame_token,
            latency_ms=(time.monotonic() - response_started_s) * 1000.0,
            is_final=True,
        )

    @staticmethod
    def _extract_stream_delta(chunk) -> str:
        choices = getattr(chunk, "choices", None)
        if not choices:
            return ""
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if isinstance(delta, dict):
            content = delta.get("content")
        else:
            content = getattr(delta, "content", None)
        return content if isinstance(content, str) else ""

    def _encode_rgb_data_url(self, image: np.ndarray) -> str:
        """Encode an RGB image as a JPEG data URL."""
        try:
            import cv2
        except ImportError as exc:
            raise VLMDetectorError("opencv-python-headless is required for VLM JPEGs") from exc

        rgb_image = np.ascontiguousarray(image, dtype=np.uint8)
        h, w = rgb_image.shape[:2]
        max_side = self._image_max_side_px
        if max_side > 0 and max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            rgb_image = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr_image,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        if not ok:
            raise VLMDetectorError("unable to JPEG-encode scene description frame")
        base64_data = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{base64_data}"

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise VLMDetectorError(
                "openai package not installed. Install with: pip install openai"
            ) from exc

        self._client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout_s,
            max_retries=0,
        )
        return self._client


class SceneDescriptionWorker(threading.Thread):
    """Background worker for parallel streamed scene descriptions."""

    def __init__(
        self,
        receiver,
        stop_event: threading.Event,
        description_cameras: Tuple[int, ...],
        base_url: str,
        api_key: str,
        model: str,
        refresh_interval_s: float,
        timeout_s: float,
        max_tokens: int,
        jpeg_quality: int,
        image_max_side_px: int,
        max_workers: Optional[int] = None,
        initially_enabled: bool = False,
        describer_factory: Optional[Callable[[], OpenAISceneDescriber]] = None,
    ) -> None:
        super().__init__(name="SceneDescriptionWorker", daemon=True)
        self._receiver = receiver
        self._app_stop_event = stop_event
        self._worker_stop_event = threading.Event()
        self._enabled_event = threading.Event()
        if initially_enabled:
            self._enabled_event.set()
        self._description_cameras = tuple(description_cameras)
        self._refresh_interval_s = max(0.1, float(refresh_interval_s))
        self._max_workers = max(
            1,
            min(
                len(self._description_cameras) or 1,
                int(max_workers or len(self._description_cameras) or 1),
            ),
        )

        self._lock = threading.Lock()
        self._descriptions_by_camera: Dict[int, SceneDescription] = {}
        self._thread_local = threading.local()
        self._describer_factory = describer_factory or (
            lambda: OpenAISceneDescriber(
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout_s=timeout_s,
                max_tokens=max_tokens,
                jpeg_quality=jpeg_quality,
                image_max_side_px=image_max_side_px,
            )
        )
        self._model = model

    def get_descriptions(self) -> Dict[int, SceneDescription]:
        """Return latest partial or final scene descriptions from all cameras."""
        with self._lock:
            return dict(self._descriptions_by_camera)

    def request_stop(self) -> None:
        self._worker_stop_event.set()

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            self._enabled_event.set()
            return
        self._enabled_event.clear()
        with self._lock:
            self._descriptions_by_camera.clear()

    def is_enabled(self) -> bool:
        return self._enabled_event.is_set()

    def _should_stop(self) -> bool:
        return self._worker_stop_event.is_set() or self._app_stop_event.is_set()

    def _wait(self, timeout_s: float) -> None:
        self._worker_stop_event.wait(timeout_s)

    @staticmethod
    def _frame_token(frame) -> Tuple[str, int, int]:
        return (
            frame.camera_serial,
            frame.sequence,
            frame.capture_received_time_ns,
        )

    def _get_describer(self) -> OpenAISceneDescriber:
        describer = getattr(self._thread_local, "describer", None)
        if describer is None:
            describer = self._describer_factory()
            self._thread_local.describer = describer
        return describer

    def _replace_description(self, description: SceneDescription) -> None:
        with self._lock:
            self._descriptions_by_camera[description.camera_id] = description

    def _mark_camera_started(self, frame) -> None:
        self._replace_description(
            SceneDescription(
                camera_id=frame.camera_id,
                description="Starting scene stream...",
                confidence=0.0,
                timestamp=time.time(),
                frame_token=self._frame_token(frame),
                latency_ms=0.0,
                is_final=False,
            )
        )

    def _stream_camera_description(self, frame) -> None:
        camera_id = frame.camera_id
        frame_token = self._frame_token(frame)
        self._mark_camera_started(frame)
        try:
            for description in self._get_describer().stream_scene(
                frame.color,
                camera_id=camera_id,
                frame_token=frame_token,
            ):
                if self._should_stop() or not self.is_enabled():
                    return
                self._replace_description(description)
        except VLMDetectorError as exc:
            if self._should_stop() or not self.is_enabled():
                return
            logger.warning("Scene description camera %d failed: %s", camera_id, exc)
            self._replace_description(
                SceneDescription(
                    camera_id=camera_id,
                    description=f"Scene description failed: {exc}",
                    confidence=0.0,
                    timestamp=time.time(),
                    frame_token=frame_token,
                    latency_ms=0.0,
                    is_final=True,
                    error=str(exc),
                )
            )
        except Exception as exc:
            if self._should_stop() or not self.is_enabled():
                return
            logger.exception("Scene description camera %d unexpected error", camera_id)
            self._replace_description(
                SceneDescription(
                    camera_id=camera_id,
                    description=f"Scene description failed: {exc}",
                    confidence=0.0,
                    timestamp=time.time(),
                    frame_token=frame_token,
                    latency_ms=0.0,
                    is_final=True,
                    error=str(exc),
                )
            )

    def _run_description_batch(self, executor: ThreadPoolExecutor, frames) -> None:
        futures = {
            executor.submit(self._stream_camera_description, frame)
            for frame in frames
        }
        while futures and not self._should_stop():
            done, futures = wait(futures, timeout=0.05, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    future.result()
                except Exception:
                    logger.exception("Scene description worker task failed")

    @staticmethod
    def _consume_finished_future(camera_id: int, future: Future) -> None:
        try:
            future.result()
        except Exception:
            logger.exception("Scene description camera %d task failed", camera_id)

    def run(self) -> None:
        next_request_s: Dict[int, float] = {
            camera_id: 0.0 for camera_id in self._description_cameras
        }
        in_flight: Dict[int, Future] = {}
        with ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="SceneDescriptionStream",
        ) as executor:
            while not self._should_stop():
                finished_camera_ids = [
                    camera_id
                    for camera_id, future in in_flight.items()
                    if future.done()
                ]
                for camera_id in finished_camera_ids:
                    self._consume_finished_future(camera_id, in_flight.pop(camera_id))

                if not self.is_enabled():
                    self._wait(0.1)
                    continue

                now_s = time.monotonic()
                due_camera_ids = [
                    camera_id
                    for camera_id in self._description_cameras
                    if camera_id not in in_flight
                    and now_s >= next_request_s.get(camera_id, 0.0)
                ]

                if due_camera_ids:
                    latest_frames = self._receiver.snapshot_latest_frames(
                        tuple(due_camera_ids)
                    )
                    for camera_id in due_camera_ids:
                        frame = latest_frames.get(camera_id)
                        if frame is None:
                            continue
                        in_flight[camera_id] = executor.submit(
                            self._stream_camera_description,
                            frame,
                        )
                        next_request_s[camera_id] = now_s + self._refresh_interval_s

                next_due_s = min(
                    (
                        due_s
                        for camera_id, due_s in next_request_s.items()
                        if camera_id not in in_flight
                    ),
                    default=now_s + 0.1,
                )
                wait_s = 0.05 if in_flight else min(0.1, max(0.01, next_due_s - now_s))
                self._wait(wait_s)

            for future in in_flight.values():
                future.cancel()
