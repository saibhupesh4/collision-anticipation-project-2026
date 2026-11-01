"""Leakage-safe image-space features for Architecture 1 smoke experiments."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")
IMAGE_WIDTH = 1280.0
IMAGE_HEIGHT = 720.0
DIAGONAL = float(np.hypot(IMAGE_WIDTH, IMAGE_HEIGHT))
GRU_FEATURES = (
    "active_person", "active_bicycle", "active_car", "active_motorcycle", "active_bus", "active_truck",
    "mean_area_person", "mean_area_car", "speed_mean_person", "speed_mean_car",
    "acceleration_mean_person", "acceleration_mean_car", "area_growth_mean_car",
    "height_growth_mean_person", "vehicle_person_min_distance_norm",
    "vehicle_person_max_closing_norm_per_sec", "vehicle_person_pairs_near_0_2",
)


def _slope(times: np.ndarray, values: np.ndarray) -> float:
    if len(times) < 2 or float(np.ptp(times)) <= 1e-9:
        return 0.0
    return float(np.polyfit(times - times[0], values, 1)[0])


def _track_kinematics(group: pd.DataFrame) -> dict[str, float]:
    group = group.sort_values("timestamp_sec")
    t = group["timestamp_sec"].to_numpy(dtype=float)
    cx = ((group["x1"] + group["x2"]) / (2.0 * IMAGE_WIDTH)).to_numpy(dtype=float)
    cy = ((group["y1"] + group["y2"]) / (2.0 * IMAGE_HEIGHT)).to_numpy(dtype=float)
    area = (((group["x2"] - group["x1"]) * (group["y2"] - group["y1"])) / (IMAGE_WIDTH * IMAGE_HEIGHT)).to_numpy(dtype=float)
    vx, vy = _slope(t, cx), _slope(t, cy)
    speed_samples = np.hypot(np.diff(cx) / np.maximum(np.diff(t), 1e-6), np.diff(cy) / np.maximum(np.diff(t), 1e-6))
    accel = np.diff(speed_samples) / np.maximum(np.diff(t[1:]), 1e-6) if len(speed_samples) >= 2 else np.asarray([0.0])
    return {
        "vx": vx,
        "vy": vy,
        "speed": float(np.hypot(vx, vy)),
        "speed_var": float(np.var(speed_samples)) if len(speed_samples) else 0.0,
        "acceleration": float(np.mean(np.abs(accel))) if len(accel) else 0.0,
        "area_growth": _slope(t, np.log(np.maximum(area, 1e-6))),
        "height_growth": _slope(t, np.log(np.maximum(((group["y2"] - group["y1"]) / IMAGE_HEIGHT).to_numpy(dtype=float), 1e-6))),
    }


def extract_features(tracks: pd.DataFrame, endpoint_sec: float, observation_sec: float = 3.0) -> tuple[dict[str, float], dict[str, float]]:
    """Return static and trajectory features using only observations at/before endpoint."""
    end = float(endpoint_sec)
    start = end - float(observation_sec)
    recent = tracks[(tracks.timestamp_sec >= start) & (tracks.timestamp_sec <= end)].copy()
    latest_all = tracks[(tracks.timestamp_sec <= end) & (tracks.timestamp_sec >= end - 0.6)].copy()
    latest = latest_all.sort_values("timestamp_sec").groupby("track_id", as_index=False).tail(1) if not latest_all.empty else latest_all

    static: dict[str, float] = {"tracks_window": float(recent.track_id.nunique()) if not recent.empty else 0.0}
    for name in CLASSES:
        current = latest[latest.class_name == name] if not latest.empty else latest
        boxes = ((current.x2 - current.x1) * (current.y2 - current.y1) / (IMAGE_WIDTH * IMAGE_HEIGHT)).to_numpy(dtype=float) if not current.empty else np.asarray([])
        static[f"active_{name}"] = float(len(current))
        static[f"mean_area_{name}"] = float(np.mean(boxes)) if len(boxes) else 0.0
        static[f"max_area_{name}"] = float(np.max(boxes)) if len(boxes) else 0.0

    # Relative vehicle/pedestrian interaction cues; no absolute image-center shortcut.
    vehicles = latest[latest.class_name.isin(("car", "bus", "truck", "motorcycle"))] if not latest.empty else latest
    people = latest[latest.class_name == "person"] if not latest.empty else latest
    distances: list[float] = []
    for _, vehicle in vehicles.iterrows():
        vx = (float(vehicle.x1) + float(vehicle.x2)) / (2.0 * IMAGE_WIDTH)
        vy = (float(vehicle.y1) + float(vehicle.y2)) / (2.0 * IMAGE_HEIGHT)
        for _, person in people.iterrows():
            px = (float(person.x1) + float(person.x2)) / (2.0 * IMAGE_WIDTH)
            py = (float(person.y1) + float(person.y2)) / (2.0 * IMAGE_HEIGHT)
            distances.append(float(np.hypot(px - vx, py - vy) / np.sqrt(2.0)))
    static["vehicle_person_min_distance_norm"] = min(distances) if distances else 1.0
    static["vehicle_person_pairs_near_0_1"] = float(sum(d <= 0.1 for d in distances))
    static["vehicle_person_pairs_near_0_2"] = float(sum(d <= 0.2 for d in distances))

    motion: dict[str, float] = {}
    per_class: dict[str, list[dict[str, float]]] = defaultdict(list)
    if not recent.empty:
        for _, group in recent.groupby("track_id"):
            if group.class_name.iloc[0] not in CLASSES:
                continue
            per_class[str(group.class_name.iloc[0])].append(_track_kinematics(group))
    for name in CLASSES:
        items = per_class[name]
        for feature in ("speed", "speed_var", "acceleration", "area_growth", "height_growth"):
            values = [x[feature] for x in items]
            motion[f"{feature}_mean_{name}"] = float(np.mean(values)) if values else 0.0
            motion[f"{feature}_max_{name}"] = float(np.max(values)) if values else 0.0

    # Closing speed is a normalized image-space relative-motion cue, not TTC.
    velocity_by_track: dict[str, dict[str, float]] = {}
    if not recent.empty:
        for track_id, group in recent.groupby("track_id"):
            velocity_by_track[str(track_id)] = _track_kinematics(group)
    closing: list[float] = []
    for _, vehicle in vehicles.iterrows():
        vehicle_v = velocity_by_track.get(str(vehicle.track_id), {"vx": 0.0, "vy": 0.0})
        vx = (float(vehicle.x1) + float(vehicle.x2)) / (2.0 * IMAGE_WIDTH)
        vy = (float(vehicle.y1) + float(vehicle.y2)) / (2.0 * IMAGE_HEIGHT)
        for _, person in people.iterrows():
            person_v = velocity_by_track.get(str(person.track_id), {"vx": 0.0, "vy": 0.0})
            px = (float(person.x1) + float(person.x2)) / (2.0 * IMAGE_WIDTH)
            py = (float(person.y1) + float(person.y2)) / (2.0 * IMAGE_HEIGHT)
            dx, dy = px - vx, py - vy
            dist = max(float(np.hypot(dx, dy)), 1e-6)
            rvx, rvy = person_v["vx"] - vehicle_v["vx"], person_v["vy"] - vehicle_v["vy"]
            closing.append(float(-(dx * rvx + dy * rvy) / dist))
    motion["vehicle_person_max_closing_norm_per_sec"] = max(closing) if closing else 0.0
    motion["vehicle_person_min_distance_norm"] = static["vehicle_person_min_distance_norm"]
    motion["vehicle_person_pairs_near_0_1"] = static["vehicle_person_pairs_near_0_1"]
    motion["vehicle_person_pairs_near_0_2"] = static["vehicle_person_pairs_near_0_2"]
    return static, motion


def make_horizon_examples(
    manifest: pd.DataFrame,
    track_dir,
    horizons=(0.5, 1.0, 1.5, 2.0, 4.0),
    observation_sec: float = 3.0,
) -> pd.DataFrame:
    """Make one event-relative example per video/horizon; exclude post-event data."""
    from pathlib import Path

    manifest = manifest.copy()
    track_dir = Path(track_dir)
    train_positive_events = manifest[(manifest.split == "train") & (manifest.label == 1)].event_timestamp.dropna().astype(float)
    if train_positive_events.empty:
        raise ValueError("Training split contains no positive event timestamps")
    negative_anchor = float(train_positive_events.median())
    output = []
    for row in manifest.itertuples(index=False):
        path = track_dir / f"{row.video_id}.parquet"
        if not path.is_file():
            continue
        tracks = pd.read_parquet(path)
        event = float(row.event_timestamp) if pd.notna(row.event_timestamp) else None
        for horizon in horizons:
            endpoint = (event - float(horizon)) if int(row.label) == 1 and event is not None else (negative_anchor - float(horizon))
            if endpoint < observation_sec or (event is not None and endpoint >= event):
                continue
            static, trajectory = extract_features(tracks, endpoint, observation_sec)
            output.append({
                "video_id": str(row.video_id),
                "label": int(row.label),
                "split": str(row.split),
                "horizon_sec": float(horizon),
                "endpoint_sec": float(endpoint),
                "observation_start_sec": float(endpoint - observation_sec),
                "event_timestamp_sec": event if event is not None else None,
                "features_static": static,
                "features_trajectory": {**static, **trajectory},
            })
    # Store dictionaries as stable JSON strings in a single table; inputs remain explicit.
    rows = []
    for item in output:
        meta = {k: v for k, v in item.items() if not k.startswith("features_")}
        rows.append({**meta, **{f"static__{k}": v for k, v in item["features_static"].items()}, **{f"trajectory__{k}": v for k, v in item["features_trajectory"].items()}})
    result = pd.DataFrame(rows)
    if not result.empty:
        feature_cols = [c for c in result if c.startswith("static__") or c.startswith("trajectory__")]
        if not np.isfinite(result[feature_cols].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite features found")
        positive = result[result.label == 1]
        if (positive.endpoint_sec >= positive.event_timestamp_sec.astype(float)).any():
            raise ValueError("Post-event observation endpoint detected")
    return result


def make_temporal_examples(
    manifest: pd.DataFrame,
    track_dir,
    horizons=(0.5, 1.0, 1.5, 2.0, 4.0),
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build six-step GRU inputs from a 3-second pre-endpoint observation."""
    from pathlib import Path

    manifest = manifest.copy()
    track_dir = Path(track_dir)
    train_events = manifest[(manifest.split == "train") & (manifest.label == 1)].event_timestamp.dropna().astype(float)
    if train_events.empty:
        raise ValueError("Training split contains no positive event timestamps")
    negative_anchor = float(train_events.median())
    metadata = []
    sequences = []
    for row in manifest.itertuples(index=False):
        path = track_dir / f"{row.video_id}.parquet"
        if not path.is_file():
            continue
        tracks = pd.read_parquet(path)
        event = float(row.event_timestamp) if pd.notna(row.event_timestamp) else None
        for horizon in horizons:
            endpoint = event - float(horizon) if int(row.label) == 1 and event is not None else negative_anchor - float(horizon)
            if endpoint < 3.0 or (event is not None and endpoint >= event):
                continue
            time_points = np.linspace(endpoint - 2.5, endpoint, num=6)
            seq = []
            for timestamp in time_points:
                static, motion = extract_features(tracks, timestamp, observation_sec=0.5)
                merged = {**static, **motion}
                seq.append([float(merged.get(name, 0.0)) for name in GRU_FEATURES])
            metadata.append({
                "video_id": str(row.video_id),
                "label": int(row.label),
                "split": str(row.split),
                "horizon_sec": float(horizon),
                "endpoint_sec": float(endpoint),
                "event_timestamp_sec": event,
            })
            sequences.append(seq)
    array = np.asarray(sequences, dtype=np.float32)
    if len(metadata):
        frame = pd.DataFrame(metadata)
        positive = frame[frame.label == 1]
        if (positive.endpoint_sec >= positive.event_timestamp_sec.astype(float)).any():
            raise ValueError("Post-event sequence endpoint detected")
        if not np.isfinite(array).all():
            raise ValueError("Non-finite GRU features found")
    else:
        frame = pd.DataFrame(columns=["video_id", "label", "split", "horizon_sec", "endpoint_sec", "event_timestamp_sec"])
    return frame, array
