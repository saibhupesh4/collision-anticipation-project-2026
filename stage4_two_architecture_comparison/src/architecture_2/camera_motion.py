"""Sparse optical-flow affine camera-motion estimate with quality flags."""

from __future__ import annotations

import cv2
import numpy as np


def estimate_camera_motion(previous_gray, current_gray, previous_boxes=()) -> dict[str, float | int | bool]:
    height, width = previous_gray.shape[:2]
    mask = np.full((height, width), 255, dtype=np.uint8)
    for box in previous_boxes:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        pad = 12
        cv2.rectangle(mask, (max(0, x1 - pad), max(0, y1 - pad)), (min(width - 1, x2 + pad), min(height - 1, y2 + pad)), 0, -1)
    points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=300, qualityLevel=0.01, minDistance=8, mask=mask, blockSize=7)
    if points is None or len(points) < 8:
        return {"valid": False, "dx_px": 0.0, "dy_px": 0.0, "scale": 1.0, "rotation_rad": 0.0, "inliers": 0, "inlier_ratio": 0.0, "uncertainty": 1.0}
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None, winSize=(21, 21), maxLevel=3)
    if next_points is None or status is None:
        return {"valid": False, "dx_px": 0.0, "dy_px": 0.0, "scale": 1.0, "rotation_rad": 0.0, "inliers": 0, "inlier_ratio": 0.0, "uncertainty": 1.0}
    good = status.ravel() == 1
    src, dst = points.reshape(-1, 2)[good], next_points.reshape(-1, 2)[good]
    if len(src) < 8:
        return {"valid": False, "dx_px": 0.0, "dy_px": 0.0, "scale": 1.0, "rotation_rad": 0.0, "inliers": 0, "inlier_ratio": 0.0, "uncertainty": 1.0}
    matrix, inlier_mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=200, confidence=0.99)
    if matrix is None or inlier_mask is None:
        return {"valid": False, "dx_px": 0.0, "dy_px": 0.0, "scale": 1.0, "rotation_rad": 0.0, "inliers": 0, "inlier_ratio": 0.0, "uncertainty": 1.0}
    inliers = int(inlier_mask.sum())
    ratio = float(inliers / max(len(src), 1))
    scale = float(np.sqrt(max(np.linalg.det(matrix[:, :2]), 0.0)))
    rotation = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    dx, dy = float(matrix[0, 2]), float(matrix[1, 2])
    valid = inliers >= 8 and ratio >= 0.2 and np.isfinite(matrix).all()
    return {
        "valid": bool(valid),
        "dx_px": dx if valid else 0.0,
        "dy_px": dy if valid else 0.0,
        "scale": scale if valid else 1.0,
        "rotation_rad": rotation if valid else 0.0,
        "inliers": inliers,
        "inlier_ratio": ratio,
        "uncertainty": float(np.clip(1.0 - ratio, 0.0, 1.0)) if valid else 1.0,
    }
