#!/usr/bin/env python3
"""OpenAI-compatible VLM object localization for RGB frames."""

from __future__ import annotations

import base64
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

QWEN_BBOX_COORD_MAX = 1000.0


VLM_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "bbox_2d": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "number"},
                    },
                },
                "required": ["label", "confidence", "bbox_2d"],
            },
        },
    },
    "required": ["objects"],
}


VLM_LOCALIZATION_PROMPT = """Locate visible physical objects in this image.
Return at most {max_objects} salient objects. Return only JSON matching this schema:
{{"objects":[{{"label":"object name","confidence":0.0,"bbox_2d":[x1,y1,x2,y2]}}]}}
Use bbox_2d coordinates normalized to the 0-1000 image coordinate range, with x increasing rightward and y increasing downward.
Use a tight rectangle around each object. Do not infer hidden objects or return scene regions."""

VLM_FAST_LINE_PROMPT = """{object_request}
Output one object per line exactly as label|confidence|x1|y1|x2|y2.
Use bbox coordinates in the 0-1000 image range. No prose.
If no visible physical object exists, output none|0|0|0|0|0."""


@dataclass(frozen=True)
class VLMDetection2D:
    """A validated semantic 2D object detection from a VLM response."""

    camera_id: int
    label: str
    confidence: float
    bbox_xyxy: np.ndarray
    frame_timestamp: float
    frame_token: Tuple[str, int, int]


class VLMDetectorError(RuntimeError):
    """Raised when the VLM request or structured response fails."""


class OpenAIVLMObjectDetector:
    """Query a vLLM OpenAI-compatible endpoint for semantic object boxes."""

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
        max_objects: int,
        conf_threshold: float,
        min_box_side_px: float,
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
        self._max_objects = max(1, int(max_objects))
        self._conf_threshold = float(conf_threshold)
        self._min_box_side_px = max(1.0, float(min_box_side_px))
        self._client = None

    def detect(
        self,
        color_image: np.ndarray,
        *,
        camera_id: int,
        frame_token: Tuple[str, int, int],
        frame_timestamp: Optional[float] = None,
    ) -> List[VLMDetection2D]:
        """Return semantic object detections for one RGB image."""
        if (
            not isinstance(color_image, np.ndarray)
            or color_image.ndim != 3
            or color_image.shape[2] != 3
            or color_image.size == 0
        ):
            raise ValueError("VLM color image must be a non-empty HxWx3 array")

        jpeg_data_url = self._encode_rgb_data_url(color_image)
        response_started = time.monotonic()
        try:
            response = self._ensure_client().chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self._fast_prompt(),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": jpeg_data_url},
                            },
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=self._max_tokens,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "structured_outputs": {
                        "regex": _fast_line_regex(self._max_objects),
                    },
                },
            )
        except Exception as exc:
            raise VLMDetectorError(f"VLM request failed: {exc}") from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise VLMDetectorError("VLM response did not include choices")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise VLMDetectorError("VLM response did not include JSON content")

        detections = parse_vlm_line_detections(
            content,
            camera_id=camera_id,
            image_shape=color_image.shape,
            frame_timestamp=time.time() if frame_timestamp is None else frame_timestamp,
            frame_token=frame_token,
            max_objects=self._max_objects,
            conf_threshold=self._conf_threshold,
            min_box_side_px=self._min_box_side_px,
        )
        logger.debug(
            "VLM camera %d request returned %d valid objects in %.1f ms",
            camera_id,
            len(detections),
            (time.monotonic() - response_started) * 1000.0,
        )
        return detections

    def _fast_prompt(self) -> str:
        if self._max_objects == 1:
            object_request = "Locate the most salient visible physical object."
        else:
            object_request = (
                f"Locate up to {self._max_objects} salient visible physical objects."
            )
        return VLM_FAST_LINE_PROMPT.format(object_request=object_request)

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise VLMDetectorError(
                "openai is not installed; install On_Receiver requirements"
            ) from exc

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout_s,
            max_retries=0,
        )
        return self._client

    def _encode_rgb_data_url(self, color_image: np.ndarray) -> str:
        try:
            import cv2
        except ImportError as exc:
            raise VLMDetectorError("opencv-python-headless is required for VLM JPEGs") from exc

        rgb_image = np.ascontiguousarray(color_image, dtype=np.uint8)
        rgb_image = self._resize_for_vlm(rgb_image, cv2)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr_image,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        if not ok:
            raise VLMDetectorError("unable to JPEG-encode VLM RGB frame")
        jpeg_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{jpeg_b64}"

    def _resize_for_vlm(self, rgb_image: np.ndarray, cv2) -> np.ndarray:
        if self._image_max_side_px <= 0:
            return rgb_image

        height, width = rgb_image.shape[:2]
        max_side = max(height, width)
        if max_side <= self._image_max_side_px:
            return rgb_image

        scale = self._image_max_side_px / float(max_side)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        return cv2.resize(
            rgb_image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )


def parse_vlm_detections(
    content: str,
    *,
    camera_id: int,
    image_shape: Tuple[int, ...],
    frame_timestamp: float,
    frame_token: Tuple[str, int, int],
    max_objects: int,
    conf_threshold: float,
    min_box_side_px: float,
) -> List[VLMDetection2D]:
    """Parse and validate the structured VLM object-box response."""
    root = _load_json_object(content)
    raw_objects = root.get("objects")
    if not isinstance(raw_objects, list):
        raise VLMDetectorError("VLM JSON root must contain an objects list")

    image_height, image_width = image_shape[:2]
    detections: List[VLMDetection2D] = []
    for raw_object in raw_objects:
        detection = _parse_object(
            raw_object,
            camera_id=camera_id,
            image_width=image_width,
            image_height=image_height,
            frame_timestamp=frame_timestamp,
            frame_token=frame_token,
            conf_threshold=conf_threshold,
            min_box_side_px=min_box_side_px,
        )
        if detection is None:
            continue
        detections.append(detection)
        if len(detections) >= max(1, int(max_objects)):
            break
    return detections


def parse_vlm_line_detections(
    content: str,
    *,
    camera_id: int,
    image_shape: Tuple[int, ...],
    frame_timestamp: float,
    frame_token: Tuple[str, int, int],
    max_objects: int,
    conf_threshold: float,
    min_box_side_px: float,
) -> List[VLMDetection2D]:
    """Parse compact regex-constrained `label|confidence|bbox` lines."""
    image_height, image_width = image_shape[:2]
    detections: List[VLMDetection2D] = []
    for raw_line in content.strip().splitlines():
        fields = [field.strip() for field in raw_line.split("|")]
        if len(fields) != 6:
            continue
        label, confidence, *bbox_fields = fields
        if label.lower() == "none":
            continue
        try:
            raw_object = {
                "label": label,
                "confidence": float(confidence),
                "bbox_2d": [float(value) for value in bbox_fields],
            }
        except ValueError:
            continue
        detection = _parse_object(
            raw_object,
            camera_id=camera_id,
            image_width=image_width,
            image_height=image_height,
            frame_timestamp=frame_timestamp,
            frame_token=frame_token,
            conf_threshold=conf_threshold,
            min_box_side_px=min_box_side_px,
        )
        if detection is None:
            continue
        detections.append(detection)
        if len(detections) >= max(1, int(max_objects)):
            break
    return detections


def _fast_line_regex(max_objects: int) -> str:
    """Return a compact vLLM structured-output regex for localization lines."""
    count = max(1, int(max_objects))
    label = r"[A-Za-z][A-Za-z0-9 _-]{0,40}"
    confidence = r"(?:0(?:\.[0-9]{1,3})?|1(?:\.0{1,3})?)"
    coord = r"(?:[0-9]{1,3}|1000)"
    line = rf"(?:{label}|none)\|{confidence}\|{coord}\|{coord}\|{coord}\|{coord}"
    if count == 1:
        return line
    return rf"{line}(?:\n{line}){{0,{count - 1}}}"


def _load_json_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VLMDetectorError(f"invalid VLM JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise VLMDetectorError("VLM JSON root must be an object")
    return parsed


def _parse_object(
    raw_object: Any,
    *,
    camera_id: int,
    image_width: int,
    image_height: int,
    frame_timestamp: float,
    frame_token: Tuple[str, int, int],
    conf_threshold: float,
    min_box_side_px: float,
) -> Optional[VLMDetection2D]:
    if not isinstance(raw_object, dict):
        return None

    label = raw_object.get("label")
    confidence = raw_object.get("confidence")
    raw_bbox = raw_object.get("bbox_2d")
    if not isinstance(label, str) or not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    if confidence < conf_threshold:
        return None

    label = " ".join(label.strip().split())
    if not label:
        return None
    label = label[:80]

    try:
        bbox = np.asarray(raw_bbox, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox]
    if x2 <= x1 or y2 <= y1:
        return None

    x_scale = float(image_width) / QWEN_BBOX_COORD_MAX
    y_scale = float(image_height) / QWEN_BBOX_COORD_MAX
    x1 = float(np.clip(x1, 0.0, QWEN_BBOX_COORD_MAX) * x_scale)
    x2 = float(np.clip(x2, 0.0, QWEN_BBOX_COORD_MAX) * x_scale)
    y1 = float(np.clip(y1, 0.0, QWEN_BBOX_COORD_MAX) * y_scale)
    y2 = float(np.clip(y2, 0.0, QWEN_BBOX_COORD_MAX) * y_scale)
    if (x2 - x1) < min_box_side_px or (y2 - y1) < min_box_side_px:
        return None

    return VLMDetection2D(
        camera_id=int(camera_id),
        label=label,
        confidence=confidence,
        bbox_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        frame_timestamp=float(frame_timestamp),
        frame_token=frame_token,
    )
