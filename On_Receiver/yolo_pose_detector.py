#!/usr/bin/env python3
"""YOLO pose estimation-based person detector.

Wraps Ultralytics YOLO pose models to detect people and their skeletal keypoints
in RGB frames. Returns clean PoseDetection2D dataclass instances with bounding boxes,
keypoints (17 COCO joints), confidence scores, and class labels.
"""

import contextlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# COCO keypoint names (17 joints)
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

# Skeleton connections as pairs of keypoint indices (COCO format)
COCO_SKELETON_CONNECTIONS = [
    (0, 1),   # nose -> left_eye
    (0, 2),   # nose -> right_eye
    (1, 3),   # left_eye -> left_ear
    (2, 4),   # right_eye -> right_ear
    (5, 6),   # left_shoulder -> right_shoulder
    (5, 7),   # left_shoulder -> left_elbow
    (6, 8),   # right_shoulder -> right_elbow
    (7, 9),   # left_elbow -> left_wrist
    (8, 10),  # right_elbow -> right_wrist
    (5, 11),  # left_shoulder -> left_hip
    (6, 12),  # right_shoulder -> right_hip
    (11, 12), # left_hip -> right_hip
    (11, 13), # left_hip -> left_knee
    (12, 14), # right_hip -> right_knee
    (13, 15), # left_knee -> left_ankle
    (14, 16), # right_knee -> right_ankle
]


@dataclass
class PoseDetection2D:
    """A single 2D person detection with pose keypoints from YOLO."""

    camera_id: int
    class_id: int  # COCO class id (0 = person)
    label: str  # "person"
    confidence: float
    bbox_xyxy: np.ndarray  # [x1, y1, x2, y2] in pixel coords (float)
    keypoints_xy: np.ndarray  # shape (K, 2) — [x, y] per keypoint in pixels
    keypoints_confidence: np.ndarray  # shape (K,) — confidence per keypoint
    frame_timestamp: float  # time.time() when frame was captured


class YoloPosePersonDetector:
    """YOLO pose detector filtered to person class only.

    Args:
        model_path: Path or name of a YOLO pose model.
            Examples: "yolo11n-pose.pt", "yolo11s-pose.pt", "yolov8n-pose.pt",
            or an exported TensorRT/ONNX model path.
        conf_threshold: Minimum confidence score for detections.
        iou_threshold: IoU threshold for NMS.
        imgsz: Inference image size (square).
        device: Device string ("", "cpu", "cuda:0", etc.). Empty = auto.
        half_precision: Use FP16 inference when on CUDA.
        target_class_id: COCO class id to keep (default 0 = person).
        batch_size: Number of camera images per GPU inference call.
        max_det: Maximum pose detections per image.
    """

    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "",
        half_precision: bool = True,
        target_class_id: int = 0,
        batch_size: int = 1,
        max_det: int = 10,
    ) -> None:
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._imgsz = imgsz
        self._device = device
        self._half_precision = half_precision
        self._target_class_id = target_class_id
        self._batch_size = max(1, int(batch_size))
        self._max_det = max(1, int(max_det))

        self._model = None
        self._loaded = False
        self._prepared = False

    def _ensure_model_loaded(self) -> None:
        """Lazily load the YOLO pose model on first use."""
        if self._loaded:
            return

        try:
            from ultralytics import YOLO

            logger.info("Loading YOLO pose model: %s", self._model_path)
            self._model = YOLO(self._model_path)

            if hasattr(self._model, "predict"):
                logger.info("YOLO pose model loaded successfully")
            else:
                logger.warning("YOLO pose model loaded but may not support predict()")

            self._loaded = True
        except ImportError:
            logger.error(
                "ultralytics is not installed. Install it with: pip install ultralytics"
            )
            raise
        except Exception as exc:
            logger.error("Failed to load YOLO pose model '%s': %s", self._model_path, exc)
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
        logger.info("Preparing YOLO pose model with a %d-image warmup", batch_size)
        self._predict_images([dummy_image for _ in range(batch_size)])
        self._prepared = True
        logger.info("YOLO pose model ready")

    def detect(
        self,
        color_image: np.ndarray,
        camera_id: int,
    ) -> List[PoseDetection2D]:
        """Run YOLO-pose on a single RGB image and return person detections with keypoints.

        Args:
            color_image: RGB image as numpy array (H, W, 3), dtype uint8.
            camera_id: Camera index for this detection.

        Returns:
            List of PoseDetection2D objects for detected persons.
            Empty list if no persons found or model is unavailable.
        """
        return self.detect_batch([(camera_id, color_image)])

    def detect_batch(
        self,
        camera_images: Sequence[Tuple[int, np.ndarray]],
    ) -> List[PoseDetection2D]:
        """Run one batched YOLO call for camera RGB images.

        Args:
            camera_images: List of (camera_id, color_image) tuples.

        Returns:
            List of PoseDetection2D objects across all cameras.
        """
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

        detections: List[PoseDetection2D] = []
        timestamp = time.time()
        for start in range(0, len(valid_images), self._batch_size):
            chunk = valid_images[start : start + self._batch_size]
            try:
                results = self._predict_images([image for _camera_id, image in chunk])
            except Exception as exc:
                logger.warning("YOLO pose GPU inference failed: %s", exc)
                continue

            for result, (camera_id, color_image) in zip(results, chunk):
                detections.extend(
                    self._detections_from_result(result, color_image, camera_id, timestamp)
                )

        return detections

    def _predict_images(self, images: Sequence[np.ndarray]):
        """Run YOLO predict on a list of images."""
        try:
            return self._run_predict(images)
        except Exception as exc:
            if self._is_cuda_oom(exc):
                logger.warning("YOLO pose CUDA OOM on %s; clearing CUDA cache", self._device or "auto")
                self._clear_cuda_cache()
            raise

    def _run_predict(self, images: Sequence[np.ndarray]):
        return self._model.predict(
            source=list(images),
            batch=max(1, min(self._batch_size, len(images))),
            imgsz=self._imgsz,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            classes=[self._target_class_id],
            device=self._device if self._device else None,
            half=self._half_precision and (self._device != "cpu"),
            max_det=self._max_det,
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
    ) -> List[PoseDetection2D]:
        """Extract PoseDetection2D objects from a YOLO result."""
        if result is None:
            return []

        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or len(boxes) == 0:
            return []

        h_orig, w_orig = color_image.shape[:2]
        detections: List[PoseDetection2D] = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id != self._target_class_id:
                continue

            conf = float(boxes.conf[i].item())
            bbox = boxes.xyxy[i].cpu().numpy().astype(np.float32)  # [x1,y1,x2,y2]

            # Extract keypoints — shape (K, 2) for xy coordinates
            kpts_xy = np.zeros((17, 2), dtype=np.float32)
            kpts_conf = np.zeros(17, dtype=np.float32)

            if keypoints is not None and i < len(keypoints):
                kpts_data = keypoints.data[i].cpu().numpy()  # (K, 3) — x, y, visibility
                num_kpts = min(kpts_data.shape[0], 17)
                kpts_xy[:num_kpts] = kpts_data[:num_kpts, :2].astype(np.float32)
                if kpts_data.shape[1] >= 3:
                    kpts_conf[:num_kpts] = kpts_data[:num_kpts, 2].astype(np.float32)

            detections.append(
                PoseDetection2D(
                    camera_id=camera_id,
                    class_id=cls_id,
                    label="person",
                    confidence=conf,
                    bbox_xyxy=bbox,
                    keypoints_xy=kpts_xy,
                    keypoints_confidence=kpts_conf,
                    frame_timestamp=timestamp,
                )
            )

        return detections
