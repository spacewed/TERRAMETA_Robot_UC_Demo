#!/usr/bin/env python3
"""YOLO segmentation detector for RGB frames.

Wraps Ultralytics YOLO instance segmentation to detect configured model classes.
Returns clean Detection2D dataclass instances with masks, bounding boxes,
confidence scores, and class labels.
"""

import contextlib
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass
class Detection2D:
    """A single 2D object detection from YOLO segmentation."""

    camera_id: int
    class_id: int  # COCO class id for pretrained YOLO11 segmentation models
    label: str
    confidence: float
    bbox_xyxy: np.ndarray  # [x1, y1, x2, y2] in pixel coords (float)
    mask: np.ndarray  # binary mask H x W (bool), same resolution as input image
    frame_timestamp: float  # time.time() when frame was captured


class YoloSegDetector:
    """YOLO segmentation detector with an optional class filter.

    Args:
        model_path: Path or name of a YOLO segmentation model.
            Examples: "yolo11n-seg.pt", "yolo11s-seg.pt", "yolov8n-seg.pt",
            or an exported TensorRT/ONNX model path.
        conf_threshold: Minimum confidence score for detections.
        iou_threshold: IoU threshold for NMS.
        imgsz: Inference image size (square).
        device: Device string ("", "cpu", "cuda:0", etc.). Empty = auto.
        half_precision: Use FP16 inference when on CUDA.
        class_ids: Model class ids to keep. None keeps every predicted class.
        batch_size: Number of camera images per GPU inference call.
        max_det: Maximum detections per image.
        retina_masks: Ask Ultralytics for full-resolution masks. Higher VRAM.
    """

    def __init__(
        self,
        model_path: str = "yolo11n-seg.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "",
        half_precision: bool = True,
        class_ids: Optional[Sequence[int]] = (0,),
        batch_size: int = 1,
        max_det: int = 20,
        retina_masks: bool = False,
    ) -> None:
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._imgsz = imgsz
        self._device = device
        self._half_precision = half_precision
        self._class_ids = self._normalize_class_ids(class_ids)
        self._batch_size = max(1, int(batch_size))
        self._max_det = max(1, int(max_det))
        self._retina_masks = bool(retina_masks)

        self._model = None
        self._loaded = False
        self._prepared = False

    def _ensure_model_loaded(self) -> None:
        """Lazily load the YOLO model on first use."""
        if self._loaded:
            return

        try:
            from ultralytics import YOLO

            logger.info("Loading YOLO segmentation model: %s", self._model_path)
            self._model = YOLO(self._model_path)

            # Warm up with a dummy inference if possible
            if hasattr(self._model, "predict"):
                logger.info("YOLO model loaded successfully")
            else:
                logger.warning("YOLO model loaded but may not support predict()")

            self._loaded = True
        except ImportError:
            logger.error(
                "ultralytics is not installed. Install it with: pip install ultralytics"
            )
            raise
        except Exception as exc:
            logger.error("Failed to load YOLO model '%s': %s", self._model_path, exc)
            raise

    def prepare(self, warmup_batch_size: int = 1) -> None:
        """Load the model and run a dummy inference on the target device."""
        if self._prepared:
            return

        self._ensure_model_loaded()
        if self._model is None:
            return

        dummy_image = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        batch_size = max(1, min(int(warmup_batch_size), self._batch_size))
        logger.info("Preparing YOLO model with a %d-image warmup", batch_size)
        self._predict_images([dummy_image for _ in range(batch_size)])
        self._prepared = True
        logger.info("YOLO model ready")

    def detect(
        self,
        color_image: np.ndarray,
        camera_id: int,
    ) -> List[Detection2D]:
        """Run YOLO-seg on a single RGB image and return detections.

        Args:
            color_image: RGB image as numpy array (H, W, 3), dtype uint8.
            camera_id: Camera index for this detection.

        Returns:
            List of Detection2D objects for configured detected classes.
            Empty list if no objects are found or model is unavailable.
        """
        return self.detect_batch([(camera_id, color_image)])

    def detect_batch(
        self,
        camera_images: Sequence[Tuple[int, np.ndarray]],
    ) -> List[Detection2D]:
        """Run one batched YOLO call for camera RGB images."""
        valid_images = [
            (camera_id, image)
            for camera_id, image in camera_images
            if image is not None and image.size > 0
        ]
        if not valid_images:
            return []

        self._ensure_model_loaded()
        if self._model is None:
            return []

        detections: List[Detection2D] = []
        timestamp = time.time()
        for start in range(0, len(valid_images), self._batch_size):
            chunk = valid_images[start : start + self._batch_size]
            try:
                results = self._predict_images([image for _camera_id, image in chunk])
            except Exception as exc:
                logger.warning("YOLO GPU inference failed: %s", exc)
                continue

            for result, (camera_id, color_image) in zip(results, chunk):
                detections.extend(
                    self._detections_from_result(result, color_image, camera_id, timestamp)
                )

        return detections

    def set_class_ids(self, class_ids: Optional[Sequence[int]]) -> None:
        """Update predicted model classes without reloading the YOLO model."""
        self._class_ids = self._normalize_class_ids(class_ids)

    @staticmethod
    def _normalize_class_ids(
        class_ids: Optional[Sequence[int]],
    ) -> Optional[Tuple[int, ...]]:
        if class_ids is None:
            return None
        return tuple(sorted({int(class_id) for class_id in class_ids}))

    def _predict_images(self, images: Sequence[np.ndarray]):
        try:
            return self._run_predict(images)
        except Exception as exc:
            if self._is_cuda_oom(exc):
                logger.warning("YOLO CUDA OOM on %s; clearing CUDA cache", self._device or "auto")
                self._clear_cuda_cache()
            raise

    def _run_predict(self, images: Sequence[np.ndarray]):
        return self._model.predict(
            source=list(images),
            batch=max(1, min(self._batch_size, len(images))),
            imgsz=self._imgsz,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            classes=list(self._class_ids) if self._class_ids is not None else None,
            device=self._device if self._device else None,
            half=self._half_precision and (self._device != "cpu"),
            max_det=self._max_det,
            retina_masks=self._retina_masks,
            verbose=False,
        )

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "cuda out of memory" in message
            or "cublas_status_alloc_failed" in message
            or "cudnn_status_alloc_failed" in message
        )

    @staticmethod
    def _clear_cuda_cache() -> None:
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                with contextlib.suppress(Exception):
                    torch.cuda.ipc_collect()

    def _detections_from_result(
        self,
        result,
        color_image: np.ndarray,
        camera_id: int,
        timestamp: float,
    ) -> List[Detection2D]:
        if result is None:
            return []

        boxes = result.boxes
        masks = result.masks
        if boxes is None or len(boxes) == 0:
            return []

        h_orig, w_orig = color_image.shape[:2]
        box_classes = boxes.cls.cpu().numpy().astype(np.int32, copy=False)
        box_confidences = boxes.conf.cpu().numpy().astype(np.float32, copy=False)
        box_bounds = boxes.xyxy.cpu().numpy().astype(np.float32, copy=False)
        mask_stack = None
        if masks is not None and len(masks) > 0:
            # Moving one stacked tensor to CPU is much cheaper in dense
            # all-object scenes than synchronizing once per instance mask.
            mask_stack = masks.data.cpu().numpy()

        detections: List[Detection2D] = []
        for i in range(len(box_classes)):
            cls_id = int(box_classes[i])
            if self._class_ids is not None and cls_id not in self._class_ids:
                continue

            conf = float(box_confidences[i])
            bbox = box_bounds[i].copy()  # [x1,y1,x2,y2]

            # Extract mask - may be None if no mask output
            mask = None
            if mask_stack is not None and i < len(mask_stack):
                mask_data = mask_stack[i]
                # masks.data is (N, H_mask, W_mask) - resize to original image size
                if mask_data.shape[0] != h_orig or mask_data.shape[1] != w_orig:
                    mask = self._resize_mask(mask_data, w_orig, h_orig)
                else:
                    mask = mask_data > 0.5

            if mask is None:
                # Fallback: create mask from bbox
                mask = np.zeros((h_orig, w_orig), dtype=bool)
                x1, y1, x2, y2 = bbox.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_orig, x2)
                y2 = min(h_orig, y2)
                mask[y1:y2, x1:x2] = True

            detections.append(
                Detection2D(
                    camera_id=camera_id,
                    class_id=cls_id,
                    label=self._class_label(result, cls_id),
                    confidence=conf,
                    bbox_xyxy=bbox,
                    mask=np.asarray(mask, dtype=bool),
                    frame_timestamp=timestamp,
                )
            )

        return detections

    def _class_label(self, result, class_id: int) -> str:
        names = getattr(result, "names", None)
        if names is None and self._model is not None:
            names = getattr(self._model, "names", None)
        if isinstance(names, dict):
            return str(names.get(class_id, f"class {class_id}"))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return f"class {class_id}"

    @staticmethod
    def _resize_mask(
        mask_small: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        """Resize a small mask to target dimensions using nearest-neighbor."""
        import cv2

        resized = cv2.resize(
            mask_small.astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized > 0.5


# Backward-compatible import name for the existing receiver path.
YoloSegPersonDetector = YoloSegDetector
