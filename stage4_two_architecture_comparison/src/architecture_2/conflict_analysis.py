"""Image-space object-to-ego-corridor conflict estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ego_corridor import corridor_overlap_probability
from .trajectory_prediction import predict_track


CLASS_WEIGHT = {"person": 1.0, "bicycle": 0.85, "motorcycle": 0.8, "car": 0.7, "bus": 0.6, "truck": 0.6}


def _camera_compensated(tracks: pd.DataFrame, camera_motion: pd.DataFrame | None) -> tuple[pd.DataFrame, float]:
    result = tracks.copy()
    if camera_motion is None or camera_motion.empty:
        result["camera_uncertainty"] = 0.5
        return result, 0.0
    camera = camera_motion.sort_values("frame_index").copy()
    camera["use_dx"] = np.where(camera.valid, camera.dx_px, 0.0)
    camera["use_dy"] = np.where(camera.valid, camera.dy_px, 0.0)
    camera["cum_dx"] = camera.use_dx.cumsum()
    camera["cum_dy"] = camera.use_dy.cumsum()
    camera["camera_uncertainty"] = np.where(camera.valid, camera.uncertainty, 1.0)
    result = result.merge(camera[["frame_index", "cum_dx", "cum_dy", "camera_uncertainty"]], on="frame_index", how="left")
    result[["cum_dx", "cum_dy"]] = result[["cum_dx", "cum_dy"]].fillna(0.0)
    result["camera_uncertainty"] = result.camera_uncertainty.fillna(1.0)
    # Stabilize only the estimated image translation. Scale/rotation are retained
    # as quality indicators; no metric ground plane is inferred.
    result["x1"] = result.x1 - result.cum_dx
    result["x2"] = result.x2 - result.cum_dx
    result["y1"] = result.y1 - result.cum_dy
    result["y2"] = result.y2 - result.cum_dy
    quality = float(camera.loc[camera.valid, "inlier_ratio"].mean()) if camera.valid.any() else 0.0
    return result, quality


def analyze_conflicts(
    tracks: pd.DataFrame,
    endpoint_sec: float,
    horizon_sec: float,
    camera_motion: pd.DataFrame | None = None,
) -> dict[str, float]:
    data, camera_quality = _camera_compensated(tracks, camera_motion)
    observed = data[(data.timestamp_sec <= endpoint_sec) & (data.timestamp_sec >= endpoint_sec - 3.0)]
    if observed.empty:
        observed = data[data.timestamp_sec <= endpoint_sec]
    latest = observed.sort_values("timestamp_sec").groupby("track_id", as_index=False).tail(1) if not observed.empty else observed
    scores: list[tuple[str, float, float]] = []
    gaps, uncertainties, speeds = [], [], []
    for track_id, group in data[data.timestamp_sec <= endpoint_sec].groupby("track_id"):
        group = group.sort_values("timestamp_sec")
        if group.empty:
            continue
        last = group.iloc[-1]
        gap = max(0.0, float(endpoint_sec) - float(last.timestamp_sec))
        if gap > 2.0:
            continue
        prediction = predict_track(group, endpoint_sec, horizon_sec)
        cls = str(last.class_name)
        if cls not in CLASS_WEIGHT:
            continue
        prob = corridor_overlap_probability(prediction["x"], prediction["y"], prediction["sigma"])
        camera_unc = float(last.get("camera_uncertainty", 1.0))
        uncertainty = float(np.clip(0.45 * prediction["sigma"] + 0.25 * prediction["residual"] + 0.2 * gap + 0.1 * camera_unc, 0.0, 1.0))
        score = float(prob * CLASS_WEIGHT[cls] * max(0.0, 1.0 - uncertainty))
        scores.append((cls, score, prob))
        gaps.append(gap)
        uncertainties.append(uncertainty)
        speeds.append(float(np.hypot(prediction["vx"], prediction["vy"])))

    person_scores = [s for cls, s, _ in scores if cls == "person"]
    vehicle_scores = [s for cls, s, _ in scores if cls != "person"]
    return {
        "rule_risk": max((s for _, s, _ in scores), default=0.0),
        "person_conflict_max": max(person_scores, default=0.0),
        "vehicle_conflict_max": max(vehicle_scores, default=0.0),
        "conflicting_tracks": float(sum(p >= 0.2 for _, _, p in scores)),
        "active_tracks": float(len(latest)),
        "uncertainty_mean": float(np.mean(uncertainties)) if uncertainties else 1.0,
        "uncertainty_max": max(uncertainties, default=1.0),
        "track_gap_max_sec": max(gaps, default=0.0),
        "predicted_image_speed_max_norm_per_sec": max(speeds, default=0.0),
        "camera_motion_quality": camera_quality,
    }
