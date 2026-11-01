"""Image-space conflict/risk features from shared tracks and camera motion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .conflict_analysis import analyze_conflicts


def _camera_summary(camera: pd.DataFrame | None, endpoint: float) -> dict[str, float]:
    if camera is None or camera.empty:
        return {"valid_fraction": 0.0, "dx_norm_per_sec": 0.0, "dy_norm_per_sec": 0.0, "motion_uncertainty": 1.0}
    recent = camera[(camera.timestamp_sec <= endpoint) & (camera.timestamp_sec >= endpoint - 3.0)]
    if recent.empty:
        return {"valid_fraction": 0.0, "dx_norm_per_sec": 0.0, "dy_norm_per_sec": 0.0, "motion_uncertainty": 1.0}
    valid = recent[recent.valid]
    duration = max(float(recent.timestamp_sec.max() - recent.timestamp_sec.min()), 0.2)
    return {
        "valid_fraction": float(recent.valid.mean()),
        "dx_norm_per_sec": float(valid.dx_px.sum() / 1280.0 / duration) if not valid.empty else 0.0,
        "dy_norm_per_sec": float(valid.dy_px.sum() / 720.0 / duration) if not valid.empty else 0.0,
        "motion_uncertainty": float(recent.uncertainty.mean()),
    }


def make_horizon_examples(manifest: pd.DataFrame, track_dir, camera_dir=None, horizons=(0.5, 1.0, 1.5, 2.0, 4.0)) -> pd.DataFrame:
    manifest = manifest.copy()
    track_dir = Path(track_dir)
    camera_dir = Path(camera_dir) if camera_dir else None
    train_events = manifest[(manifest.split == "train") & (manifest.label == 1)].event_timestamp.dropna().astype(float)
    if train_events.empty:
        raise ValueError("Training split contains no positive event timestamps")
    negative_anchor = float(train_events.median())
    rows = []
    for row in manifest.itertuples(index=False):
        tracks_path = track_dir / f"{row.video_id}.parquet"
        if not tracks_path.is_file():
            continue
        tracks = pd.read_parquet(tracks_path)
        camera = None
        if camera_dir is not None and (camera_dir / f"{row.video_id}.parquet").is_file():
            camera = pd.read_parquet(camera_dir / f"{row.video_id}.parquet")
        event = float(row.event_timestamp) if pd.notna(row.event_timestamp) else None
        for horizon in horizons:
            endpoint = event - float(horizon) if int(row.label) == 1 and event is not None else negative_anchor - float(horizon)
            if endpoint < 3.0 or (event is not None and endpoint >= event):
                continue
            compensated = analyze_conflicts(tracks, endpoint, float(horizon), camera)
            no_camera = analyze_conflicts(tracks, endpoint, float(horizon), None)
            camera_features = _camera_summary(camera, endpoint)
            item = {
                "video_id": str(row.video_id),
                "label": int(row.label),
                "split": str(row.split),
                "horizon_sec": float(horizon),
                "endpoint_sec": float(endpoint),
                "event_timestamp_sec": event,
            }
            item.update({f"comp__{k}": float(v) for k, v in compensated.items()})
            item.update({f"no_camera__{k}": float(v) for k, v in no_camera.items()})
            item.update({f"camera__{k}": float(v) for k, v in camera_features.items()})
            rows.append(item)
    result = pd.DataFrame(rows)
    if not result.empty:
        positive = result[result.label == 1]
        if (positive.endpoint_sec >= positive.event_timestamp_sec.astype(float)).any():
            raise ValueError("Post-event endpoint detected in Architecture 2 examples")
        features = [c for c in result if c.startswith(("comp__", "no_camera__", "camera__"))]
        if not np.isfinite(result[features].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite Architecture 2 features found")
    return result
