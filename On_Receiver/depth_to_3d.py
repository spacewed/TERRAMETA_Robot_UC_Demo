#!/usr/bin/env python3
"""Convert YOLO segmentation masks + depth images into 3D bounding boxes.

Takes a depth image and a binary mask from YOLO segmentation, extracts valid
depth pixels under the mask, back-projects them into 3D using camera intrinsics,
applies camera rotation/offset to world coordinates, and fits an axis-aligned
bounding box around the resulting point cloud.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MIN_TRIMMED_EXTENT_POINTS = 20


@dataclass
class PersonBBox3D:
    """An axis-aligned 3D bounding box for a detected object.

    All coordinates are in metres, in the world coordinate frame
    (after camera rotation and offset have been applied).
    """

    camera_id: int
    center_xyz: np.ndarray  # [X, Y, Z] centre of the box in world coords
    size_xyz: np.ndarray  # [width, height, depth] extents in metres
    confidence: float
    label: str  # e.g. "person", "car", "chair"
    num_points: int  # number of valid depth points used
    timestamp: float
    class_id: int = 0  # COCO class id for per-class color coding
    source: str = "yolo"  # "yolo", "vlm", etc. for rendering/tracking


def fit_person_bbox(
    depth_image: np.ndarray,
    mask: np.ndarray,
    camera_id: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    camera_rotation_matrix: np.ndarray,
    camera_offset: np.ndarray,
    bbox_xyxy: Optional[np.ndarray] = None,
    max_depth_m: float = 10.0,
    min_depth_m: float = 0.1,
    erosion_kernel: int = 3,
    subsample_step: int = 2,
    outlier_threshold_m: float = 0.6,
    extent_trim_percentile: float = 1.0,
    min_points: int = 10,
    confidence: float = 0.0,
    label: str = "person",
    timestamp: float = 0.0,
    class_id: int = 0,
    source: str = "yolo",
) -> Optional[PersonBBox3D]:
    """Fit an axis-aligned 3D bounding box from a masked depth region.

    Args:
        depth_image: Depth image in millimetres (uint16), shape (H, W).
        mask: Binary mask (bool), shape (H, W). True where person pixels are.
        camera_id: Camera index.
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.
        camera_rotation_matrix: 3x3 rotation matrix from camera-local to world.
        camera_offset: 3-element translation offset from camera-local to world.
        bbox_xyxy: Optional 2D detection bounds used to crop mask/depth work.
        max_depth_m: Maximum valid depth in metres.
        min_depth_m: Minimum valid depth in metres.
        erosion_kernel: Kernel size for morphological erosion (0 = disabled).
        subsample_step: Subsample every Nth pixel for speed (1 = all pixels).
        outlier_threshold_m: Keep points within this distance of median depth.
        extent_trim_percentile: Trim this percentile from each 3D extent end
            after depth outlier filtering (0 = raw min/max).
        min_points: Minimum valid points required to produce a bbox.
        confidence: Confidence score from detection.
        label: Label string.
        timestamp: Frame timestamp.
        class_id: COCO class id for per-class color coding.
        source: Detection source used by renderers and label tracking.

    Returns:
        PersonBBox3D if enough valid points exist, else None.
    """
    if depth_image is None or mask is None:
        return None

    h, w = depth_image.shape[:2]
    mh, mw = mask.shape[:2]

    # Ensure mask matches depth resolution
    if mh != h or mw != w:
        import cv2

        mask = cv2.resize(
            mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    # The mask remains authoritative within the detection ROI. Cropping here
    # avoids full-frame morphology and coordinate extraction for each person.
    depth_roi = depth_image
    mask_roi = mask
    x_offset = 0
    y_offset = 0
    if bbox_xyxy is not None and len(bbox_xyxy) >= 4:
        pad = max(1, erosion_kernel)
        x1, y1, x2, y2 = np.asarray(bbox_xyxy[:4], dtype=np.float64)
        left = max(0, int(np.floor(x1)) - pad)
        top = max(0, int(np.floor(y1)) - pad)
        right = min(w, int(np.ceil(x2)) + pad)
        bottom = min(h, int(np.ceil(y2)) + pad)
        if right <= left or bottom <= top:
            return None
        depth_roi = depth_image[top:bottom, left:right]
        mask_roi = mask[top:bottom, left:right]
        x_offset = left
        y_offset = top

    # Erode mask to reduce edge/background leakage
    if erosion_kernel > 0:
        import cv2

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erosion_kernel, erosion_kernel)
        )
        mask_roi = cv2.erode(mask_roi.astype(np.uint8), kernel).astype(bool)

    # Extract pixel coordinates where mask is True
    ys_roi, xs_roi = np.where(mask_roi)
    if len(xs_roi) == 0:
        return None

    # Subsample for speed
    if subsample_step > 1:
        indices = np.arange(0, len(xs_roi), subsample_step)
        xs_roi = xs_roi[indices]
        ys_roi = ys_roi[indices]

    # Sample depth values at mask pixels
    depth_mm = depth_roi[ys_roi, xs_roi].astype(np.float64)

    # Filter invalid depths
    valid = (
        (depth_mm > 0)
        & (~np.isnan(depth_mm))
        & (depth_mm >= min_depth_m * 1000.0)
        & (depth_mm <= max_depth_m * 1000.0)
    )

    if valid.sum() < min_points:
        return None

    depth_valid = depth_mm[valid]
    xs_valid = xs_roi[valid] + x_offset
    ys_valid = ys_roi[valid] + y_offset

    # Compute median depth for robustness
    median_depth_m = float(np.median(depth_valid)) / 1000.0

    # Reject outliers: keep points within ±outlier_threshold_m of median
    depth_m = depth_valid / 1000.0
    within_range = np.abs(depth_m - median_depth_m) <= outlier_threshold_m
    if within_range.sum() < min_points:
        return None

    depth_final = depth_m[within_range]
    xs_final = xs_valid[within_range]
    ys_final = ys_valid[within_range]

    num_points = int(len(depth_final))

    # Back-project to 3D in camera-local coordinates
    X_local = ((xs_final - cx) * depth_final) / fx
    Y_local = -((ys_final - cy) * depth_final) / fy  # negate Y for OpenGL convention
    Z_local = depth_final

    points_cam = np.stack([X_local, Y_local, Z_local], axis=-1)  # (N, 3)

    # Transform to world coordinates.
    # The GPU shader receives the rotation matrix in column-major (OpenGL)
    # format, which effectively transposes it. We must match that here.
    points_world = (camera_rotation_matrix.T @ points_cam.T).T + camera_offset

    # Fit robust extents after depth filtering. Pixel-aligned segmentation masks
    # still carry occasional edge and depth-noise points that can stretch raw
    # min/max boxes far beyond the visible person.
    mins, maxs = _fit_axis_aligned_extents(
        points_world,
        trim_percentile=extent_trim_percentile,
    )
    center = (mins + maxs) / 2.0
    size = maxs - mins

    # Ensure minimum size (a person should be at least ~0.3m wide/tall/deep)
    size = np.maximum(size, 0.3)

    return PersonBBox3D(
        camera_id=camera_id,
        center_xyz=center.astype(np.float32),
        size_xyz=size.astype(np.float32),
        confidence=confidence,
        label=label,
        num_points=num_points,
        timestamp=timestamp,
        class_id=class_id,
        source=source,
    )


def fit_bbox_only(
    depth_image: np.ndarray,
    bbox_xyxy: np.ndarray,
    camera_id: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    camera_rotation_matrix: np.ndarray,
    camera_offset: np.ndarray,
    max_depth_m: float = 10.0,
    min_depth_m: float = 0.1,
    erosion_kernel: int = 3,
    subsample_step: int = 2,
    outlier_threshold_m: float = 0.6,
    extent_trim_percentile: float = 1.0,
    min_points: int = 10,
    confidence: float = 0.0,
    label: str = "object",
    timestamp: float = 0.0,
    source: str = "vlm",
    class_id: int = -1,
) -> Optional[PersonBBox3D]:
    """Fit a 3D object box from a rectangular 2D RGB/depth ROI."""
    if depth_image is None or bbox_xyxy is None or len(bbox_xyxy) < 4:
        return None

    height, width = depth_image.shape[:2]
    x1, y1, x2, y2 = np.asarray(bbox_xyxy[:4], dtype=np.float64)
    left = max(0, int(np.floor(x1)))
    top = max(0, int(np.floor(y1)))
    right = min(width, int(np.ceil(x2)))
    bottom = min(height, int(np.ceil(y2)))
    if right <= left or bottom <= top:
        return None

    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return fit_person_bbox(
        depth_image=depth_image,
        mask=mask,
        camera_id=camera_id,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        camera_rotation_matrix=camera_rotation_matrix,
        camera_offset=camera_offset,
        bbox_xyxy=np.asarray([left, top, right, bottom], dtype=np.float32),
        max_depth_m=max_depth_m,
        min_depth_m=min_depth_m,
        erosion_kernel=erosion_kernel,
        subsample_step=subsample_step,
        outlier_threshold_m=outlier_threshold_m,
        extent_trim_percentile=extent_trim_percentile,
        min_points=min_points,
        confidence=confidence,
        label=label,
        timestamp=timestamp,
        class_id=class_id,
        source=source,
    )


def _fit_axis_aligned_extents(
    points_xyz: np.ndarray,
    *,
    trim_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return visible 3D extents, optionally trimming sparse edge outliers."""
    points = np.asarray(points_xyz)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("expected non-empty Nx3 points")

    trim = float(np.clip(trim_percentile, 0.0, 49.0))
    if trim <= 0.0 or points.shape[0] < MIN_TRIMMED_EXTENT_POINTS:
        return points.min(axis=0), points.max(axis=0)

    mins, maxs = np.percentile(points, (trim, 100.0 - trim), axis=0)
    return mins, maxs


def build_camera_intrinsics_from_fov(
    width: int,
    height: int,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> dict:
    """Derive approximate camera intrinsics from FOV angles.

    This mirrors the intrinsics computation used in the vertex shader
    of receiver.py.

    Returns:
        Dict with keys 'fx', 'fy', 'cx', 'cy'.
    """
    fx = width / (2.0 * np.tan(np.radians(horizontal_fov_deg) / 2.0))
    fy = height / (2.0 * np.tan(np.radians(vertical_fov_deg) / 2.0))
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
