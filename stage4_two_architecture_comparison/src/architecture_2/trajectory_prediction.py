"""Track prediction with actual timestamps and uncertainty estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .kalman_filter import ConstantVelocityKalman


def predict_track(group: pd.DataFrame, endpoint_sec: float, horizon_sec: float) -> dict[str, float | bool]:
    group = group[group.timestamp_sec <= endpoint_sec].sort_values("timestamp_sec")
    if group.empty:
        return {"valid": False, "x": 0.5, "y": 0.5, "vx": 0.0, "vy": 0.0, "sigma": 0.5, "gap_sec": float("inf"), "residual": 0.5}
    timestamps = group.timestamp_sec.to_numpy(dtype=float)
    x = ((group.x1 + group.x2) / (2 * 1280.0)).to_numpy(dtype=float)
    y = (group.y2 / 720.0).to_numpy(dtype=float)  # bottom-center for road users
    gap = max(0.0, float(endpoint_sec) - timestamps[-1])
    if len(group) < 2 or timestamps[-1] - timestamps[0] <= 1e-6:
        return {"valid": False, "x": float(x[-1]), "y": float(y[-1]), "vx": 0.0, "vy": 0.0, "sigma": min(0.5, 0.12 + gap * 0.15), "gap_sec": gap, "residual": 0.12}

    filt = ConstantVelocityKalman()
    filt.initialize(x[0], y[0])
    residuals = []
    for i in range(1, len(timestamps)):
        state, _ = filt.predict(timestamps[i] - timestamps[i - 1])
        residuals.append(float(np.hypot(x[i] - state[0], y[i] - state[1])))
        filt.update(x[i], y[i])
    state, covariance = filt.predict(max(float(horizon_sec) + gap, 1e-6))
    residual = float(np.median(residuals)) if residuals else 0.12
    # Conservative image-space uncertainty; this is not a calibrated probability.
    sigma = float(np.clip(residual + 0.025 + gap * 0.12 + np.sqrt(max(covariance[0, 0], covariance[1, 1])) * 0.1, 0.02, 0.5))
    return {"valid": True, "x": float(state[0]), "y": float(state[1]), "vx": float(state[2]), "vy": float(state[3]), "sigma": sigma, "gap_sec": gap, "residual": residual}
