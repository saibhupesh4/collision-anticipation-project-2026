"""Broad heuristic ego corridor in normalized image coordinates only."""

from __future__ import annotations

import math


def corridor_bounds(y_norm: float) -> tuple[float, float] | None:
    """Return a broad trapezoid at y; no camera calibration is implied."""
    y = float(y_norm)
    if y < 0.35 or y > 1.1:
        return None
    alpha = min(1.0, max(0.0, (y - 0.35) / 0.65))
    half_width = 0.10 + 0.22 * alpha
    return 0.5 - half_width, 0.5 + half_width


def corridor_overlap_probability(x_norm: float, y_norm: float, sigma: float) -> float:
    bounds = corridor_bounds(y_norm)
    if bounds is None:
        return 0.0
    left, right = bounds
    s = max(float(sigma), 1e-4)
    cdf = lambda z: 0.5 * (1.0 + math.erf((z - float(x_norm)) / (s * math.sqrt(2.0))))
    return float(max(0.0, min(1.0, cdf(right) - cdf(left))))


def image_space_geometry_status() -> dict[str, object]:
    return {"mode": "image_space_heuristic", "metric_geometry_valid": False, "distance_m": None, "ttc_s": None}
