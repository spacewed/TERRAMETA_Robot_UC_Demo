#!/usr/bin/env python3
"""Convert YOLO pose keypoints + depth images into 3D keypoints and bounding boxes.

Takes a depth image and 2D keypoints from YOLO pose estimation, samples depth at
each keypoint location, back-projects them into 3D using camera intrinsics,
applies camera rotation/offset to world coordinates, and fits an axis-aligned
bounding box around the resulting 3D skeleton.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from yolo_pose_detector import COCO_SKELETON_CONNECTIONS

logger = logging.getLogger(__name__)


@dataclass
class PersonPose3D:
    """A detected person with 3D keypoints and bounding box.

    All coordinates are in metres, in the world coordinate frame
    (after camera rotation and offset have been applied).
    """

    camera_id: int
    center_xyz: np.ndarray  # [X, Y, Z] centre of the box in world coords
    size_xyz: np.ndarray  # [width, height, depth] extents in metres
    confidence: float
    label: str  # "person"
    num_points: int  # number of valid depth points used for bbox
    timestamp: float
    keypoints_3d: np.ndarray  # shape (17, 3) — [X, Y, Z] per keypoint in world coords
    keypoints_confidence: np.ndarray  # shape (17,) — confidence per keypoint


def _sample_depth_neighborhood(
    depth_image: np.ndarray,
    u: int,
    v: int,
    radius: int = 3,
) -> float:
    """Sample median depth from a small neighborhood around a pixel.

    Uses a (2*radius+1)^2 patch centered on (u, v). Returns NaN if no valid
    pixels found.

    Args:
        depth_image: Depth image in millimetres (uint16), shape (H, W).
        u: Column coordinate.
        v: Row coordinate.
        radius: Half-size of the sampling patch.

    Returns:
        Median depth in millimetres, or NaN if no valid samples.
    """
    h, w = depth_image.shape[:2]
    y1 = max(0, v - radius)
    y2 = min(h, v + radius + 1)
    x1 = max(0, u - radius)
    x2 = min(w, u + radius + 1)

    patch = depth_image[y1:y2, x1:x2].astype(np.float64)
    valid = patch[(patch > 0) & (~np.isnan(patch))]
    if len(valid) == 0:
        return float("nan")
    return float(np.median(valid))


def backproject_keypoints(
    keypoints_xy: np.ndarray,
    keypoints_confidence: np.ndarray,
    depth_image: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    camera_rotation_matrix: np.ndarray,
    camera_offset: np.ndarray,
    max_depth_m: float = 10.0,
    min_depth_m: float = 0.1,
    confidence_threshold: float = 0.5,
    depth_sample_radius: int = 3,
) -> tuple:
    """Back-project 2D keypoints to 3D using depth image and camera intrinsics.

    Args:
        keypoints_xy: (K, 2) array of [x, y] pixel coordinates for each keypoint.
        keypoints_confidence: (K,) array of confidence scores per keypoint.
        depth_image: Depth image in millimetres (uint16), shape (H, W).
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.
        camera_rotation_matrix: 3x3 rotation matrix from camera-local to world.
        camera_offset: 3-element translation offset from camera-local to world.
        max_depth_m: Maximum valid depth in metres.
        min_depth_m: Minimum valid depth in metres.
        confidence_threshold: Minimum keypoint confidence to consider.
        depth_sample_radius: Radius for median depth sampling around each keypoint.

    Returns:
        Tuple of (keypoints_3d, valid_mask) where:
            keypoints_3d: (K, 3) array of [X, Y, Z] in world coords (NaN for invalid).
            valid_mask: (K,) boolean array indicating which keypoints are valid.
    """
    h, w = depth_image.shape[:2]
    K = keypoints_xy.shape[0]

    keypoints_3d = np.full((K, 3), np.nan, dtype=np.float32)
    valid_mask = np.zeros(K, dtype=bool)

    for k in range(K):
        if keypoints_confidence[k] < confidence_threshold:
            continue

        u, v = int(round(keypoints_xy[k, 0])), int(round(keypoints_xy[k, 1]))
        if u < 0 or u >= w or v < 0 or v >= h:
            continue

        # Sample median depth from neighborhood for robustness
        depth_mm = _sample_depth_neighborhood(depth_image, u, v, radius=depth_sample_radius)

        # Filter invalid depths
        if np.isnan(depth_mm) or depth_mm <= 0:
            continue
        depth_m = depth_mm / 1000.0
        if depth_m < min_depth_m or depth_m > max_depth_m:
            continue

        # Back-project to camera-local 3D
        X_local = (u - cx) * depth_m / fx
        Y_local = -(v - cy) * depth_m / fy  # negate Y for OpenGL convention
        Z_local = depth_m

        # Transform to world coordinates
        local_point = np.array([X_local, Y_local, Z_local], dtype=np.float32)
        # The GPU shader receives the rotation matrix in column-major (OpenGL)
        # format, which effectively transposes it. We must match that here.
        world_point = camera_rotation_matrix.T @ local_point + camera_offset

        keypoints_3d[k] = world_point
        valid_mask[k] = True

    return keypoints_3d, valid_mask


def fit_pose_bbox(
    depth_image: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_confidence: np.ndarray,
    camera_id: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    camera_rotation_matrix: np.ndarray,
    camera_offset: np.ndarray,
    max_depth_m: float = 10.0,
    min_depth_m: float = 0.1,
    confidence_threshold: float = 0.5,
    min_valid_keypoints: int = 4,
    min_valid_connections: int = 0,
    confidence: float = 0.0,
    label: str = "person",
    timestamp: float = 0.0,
) -> Optional[PersonPose3D]:
    """Fit a 3D bounding box from pose keypoints and depth.

    Args:
        depth_image: Depth image in millimetres (uint16), shape (H, W).
        keypoints_xy: (K, 2) array of [x, y] pixel coordinates.
        keypoints_confidence: (K,) array of confidence scores.
        camera_id: Camera index.
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.
        camera_rotation_matrix: 3x3 rotation matrix from camera-local to world.
        camera_offset: 3-element translation offset from camera-local to world.
        max_depth_m: Maximum valid depth in metres.
        min_depth_m: Minimum valid depth in metres.
        confidence_threshold: Minimum keypoint confidence to consider.
        min_valid_keypoints: Minimum number of valid keypoints required.
        min_valid_connections: Minimum number of depth-valid COCO skeleton edges.
        confidence: Confidence score from detection.
        label: Label string.
        timestamp: Frame timestamp.

    Returns:
        PersonPose3D if enough valid keypoints exist, else None.
    """
    if depth_image is None or keypoints_xy is None:
        return None

    # Use full-frame depth for back-projection to avoid intrinsics mismatch.
    # Keypoint coordinates remain in full-frame pixel space.
    # This avoids the critical issue where cropping changes the principal point.
    keypoints_3d, valid_mask = backproject_keypoints(
        keypoints_xy=keypoints_xy,
        keypoints_confidence=keypoints_confidence,
        depth_image=depth_image,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        camera_rotation_matrix=camera_rotation_matrix,
        camera_offset=camera_offset,
        max_depth_m=max_depth_m,
        min_depth_m=min_depth_m,
        confidence_threshold=confidence_threshold,
    )

    num_valid = int(valid_mask.sum())
    if num_valid < min_valid_keypoints:
        return None
    num_valid_connections = sum(
        1
        for start_idx, end_idx in COCO_SKELETON_CONNECTIONS
        if start_idx < len(valid_mask)
        and end_idx < len(valid_mask)
        and valid_mask[start_idx]
        and valid_mask[end_idx]
    )
    if num_valid_connections < min_valid_connections:
        return None

    # Fit axis-aligned bounding box from valid 3D keypoints
    valid_points = keypoints_3d[valid_mask]
    mins = valid_points.min(axis=0)
    maxs = valid_points.max(axis=0)
    center = (mins + maxs) / 2.0
    size = maxs - mins

    # Ensure minimum size (a person should be at least ~0.3m wide/tall/deep)
    size = np.maximum(size, 0.3)

    return PersonPose3D(
        camera_id=camera_id,
        center_xyz=center.astype(np.float32),
        size_xyz=size.astype(np.float32),
        confidence=confidence,
        label=label,
        num_points=num_valid,
        timestamp=timestamp,
        keypoints_3d=keypoints_3d.astype(np.float32),
        keypoints_confidence=keypoints_confidence.astype(np.float32),
    )
