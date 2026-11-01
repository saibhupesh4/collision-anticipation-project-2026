"""Resumable full-frame YOLOv8n, ByteTrack, and image-space feature pipeline.

Run from the project root with the existing collision_env Python interpreter.
This script preprocesses data only; it does not fit or select models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from tqdm.auto import tqdm


CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLES = {"car", "bus", "truck", "motorcycle"}
DET_FIELDS = [
    "video_id", "frame_index", "timestamp_sec", "track_id", "bytetrack_id", "class_id", "class_name",
    "x1", "y1", "x2", "y2", "confidence", "track_age_sec", "time_since_last_detection_sec", "track_status",
]
TRACK_FEATURE_FIELDS = [
    "video_id", "frame_index", "timestamp_sec", "track_id", "class_id", "class_name",
    "x1", "y1", "x2", "y2", "confidence", "center_x_norm", "center_y_norm",
    "width_norm", "height_norm", "area_norm", "image_vx_norm_per_sec", "image_vy_norm_per_sec",
    "image_acceleration_norm_per_sec2", "image_direction_rad", "image_direction_change_rad",
    "log_bbox_expansion_rate_per_sec", "lateral_image_movement_norm_per_sec", "kinematics_valid",
]
FRAME_FIELDS = ["video_id", "frame_index", "timestamp_sec", "detection_count", "inference_ms"]
INTERACTION_FIELDS = [
    "video_id", "frame_index", "timestamp_sec", "person_track_id", "nearest_vehicle_track_id",
    "nearest_vehicle_class", "interaction_valid", "relative_x_norm", "relative_y_norm",
    "relative_vx_norm_per_sec", "relative_vy_norm_per_sec", "proximity_norm_diagonal",
    "closing_rate_norm_per_sec", "time_to_conflict_image_sec", "within_0_05_norm", "within_0_10_norm",
    "within_0_20_norm",
]
SUMMARY_FIELDS = [
    "video_id", "category", "label", "split", "video_path", "status", "error", "fps", "expected_frames",
    "processed_frames", "width", "height", "duration_seconds", "detection_rows", "tracked_rows",
    "unique_track_ids", "trajectory_feature_rows", "interaction_feature_rows", "processing_seconds",
    "inference_seconds", "inference_batch_size", "inference_half_precision", "peak_gpu_memory_bytes", "peak_gpu_reserved_bytes", "detections_path", "frame_index_path", "tracks_path", "trajectories_path",
    "interactions_path", "completed_utc",
]
SPLITS = ("train", "validation", "test")


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".writing.parquet")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".writing.csv")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def make_split_outputs(root: Path, source_manifest: Path, seed: int) -> pd.DataFrame:
    """Reuse and verify the existing deterministic split proposal; never touch source metadata."""
    manifest = pd.read_csv(source_manifest, dtype={"video_id": str})
    required = {"video_id", "video_path", "label", "split"}
    absent = required - set(manifest.columns)
    if absent:
        raise ValueError(f"Split manifest missing columns: {sorted(absent)}")
    if manifest.video_id.duplicated().any():
        raise ValueError("Duplicate video IDs in input split manifest")
    if len(manifest) != 1500 or manifest.video_id.nunique() != 1500:
        raise ValueError(f"Expected 1,500 unique training videos, found {len(manifest)}")
    if set(manifest.split) - set(SPLITS):
        raise ValueError("Unexpected split name in existing manifest")
    # Expected fixed-seed stratified allocation: 525/112/113 from each class.
    expected = {"train": {0: 525, 1: 525}, "validation": {0: 112, 1: 112}, "test": {0: 113, 1: 113}}
    for split, labels in expected.items():
        for label, count in labels.items():
            observed = int(((manifest.split == split) & (manifest.label == label)).sum())
            if observed != count:
                raise ValueError(f"Existing split allocation differs from expected seed-{seed} counts: {split}/{label}={observed}, expected {count}")
    if manifest.video_path.map(lambda value: Path(value).is_file()).sum() != 1500:
        raise FileNotFoundError("The split manifest does not resolve to all 1,500 local MP4 files")
    out = root / "data" / "splits"
    out.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(manifest, out / "split_manifest.csv")
    for split in SPLITS:
        part = manifest[manifest.split == split].copy().sort_values(["label", "video_id"])
        write_csv_atomic(part, out / f"{split}_manifest.csv")
        (out / f"{split}_video_ids.txt").write_text("".join(f"{vid}\n" for vid in part.video_id), encoding="utf-8")
    atomic_json({"seed": seed, "split_ratio_requested": [0.70, 0.15, 0.15], "counts": {
        split: {"negative": int(((manifest.split == split) & (manifest.label == 0)).sum()),
                "positive": int(((manifest.split == split) & (manifest.label == 1)).sum())}
        for split in SPLITS}, "source_manifest": str(source_manifest),
        "source_session_grouping_available": False}, out / "split_config.json")
    return manifest


def make_stratified_subset(manifest: pd.DataFrame, positive_count: int, negative_count: int, seed: int, output_dir: Path) -> pd.DataFrame:
    """Select deterministic subsets within existing class and split assignments."""
    import random

    rng = random.Random(seed)
    selected_indices: list[int] = []
    for label, count in ((1, positive_count), (0, negative_count)):
        targets = {"train": int(count * 0.70), "validation": int(count * 0.15)}
        targets["test"] = count - targets["train"] - targets["validation"]
        for split in SPLITS:
            pool = manifest[(manifest.label == label) & (manifest.split == split)].sort_values("video_id").index.tolist()
            rng.shuffle(pool)
            if targets[split] > len(pool):
                raise ValueError(f"Not enough class={label} videos in {split} for requested subset")
            selected_indices.extend(pool[: targets[split]])
    subset = manifest.loc[selected_indices].copy().sort_values(["split", "label", "video_id"])
    if int((subset.label == 1).sum()) != positive_count or int((subset.label == 0).sum()) != negative_count:
        raise ValueError("Subset class counts do not match request")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(subset, output_dir / "split_manifest.csv")
    for split in SPLITS:
        part = subset[subset.split == split].copy().sort_values(["label", "video_id"])
        write_csv_atomic(part, output_dir / f"{split}_manifest.csv")
        (output_dir / f"{split}_video_ids.txt").write_text("".join(f"{v}\n" for v in part.video_id), encoding="utf-8")
    atomic_json({"seed": seed, "positive_videos": positive_count, "negative_videos": negative_count,
                 "source_split_manifest": "data/splits/split_manifest.csv",
                 "split_counts": {s: {"positive": int(((subset.split == s) & (subset.label == 1)).sum()),
                                       "negative": int(((subset.split == s) & (subset.label == 0)).sum())} for s in SPLITS}},
                output_dir / "subset_config.json")
    return subset


def wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def derive_features(video_id: str, tracked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive object motion and nearest vehicle/person interactions in image space."""
    if tracked.empty:
        return pd.DataFrame(columns=TRACK_FEATURE_FIELDS), pd.DataFrame(columns=INTERACTION_FIELDS)
    rows: list[dict[str, Any]] = []
    for track_id, group in tracked.groupby("track_id", sort=False):
        group = group.sort_values(["timestamp_sec", "frame_index"]).reset_index(drop=True)
        last_dir: float | None = None
        last_v: tuple[float, float] | None = None
        for i, r in enumerate(group.itertuples(index=False)):
            w = max(0.0, float(r.x2) - float(r.x1))
            h = max(0.0, float(r.y2) - float(r.y1))
            cx = (float(r.x1) + float(r.x2)) / (2.0 * 1280.0)
            cy = (float(r.y1) + float(r.y2)) / (2.0 * 720.0)
            width_norm, height_norm = w / 1280.0, h / 720.0
            area = width_norm * height_norm
            vx = vy = acceleration = direction = direction_change = expansion = np.nan
            if i > 0:
                prev = group.iloc[i - 1]
                dt = float(r.timestamp_sec) - float(prev.timestamp_sec)
                if dt > 0:
                    prev_cx = (float(prev.x1) + float(prev.x2)) / (2.0 * 1280.0)
                    prev_cy = (float(prev.y1) + float(prev.y2)) / (2.0 * 720.0)
                    vx, vy = (cx - prev_cx) / dt, (cy - prev_cy) / dt
                    direction = float(math.atan2(vy, vx))
                    if last_dir is not None:
                        direction_change = wrap_angle(direction - last_dir)
                    if last_v is not None:
                        acceleration = math.hypot(vx - last_v[0], vy - last_v[1]) / dt
                    prev_w = max(1e-12, float(prev.x2) - float(prev.x1)) / 1280.0
                    prev_h = max(1e-12, float(prev.y2) - float(prev.y1)) / 720.0
                    prev_area = max(prev_w * prev_h, 1e-12)
                    expansion = (math.log(max(area, 1e-12)) - math.log(prev_area)) / dt
                    last_dir, last_v = direction, (vx, vy)
            rows.append({
                "video_id": video_id, "frame_index": int(r.frame_index), "timestamp_sec": float(r.timestamp_sec),
                "track_id": str(track_id), "class_id": int(r.class_id), "class_name": str(r.class_name),
                "x1": float(r.x1), "y1": float(r.y1), "x2": float(r.x2), "y2": float(r.y2),
                "confidence": float(r.confidence), "center_x_norm": cx, "center_y_norm": cy,
                "width_norm": width_norm, "height_norm": height_norm, "area_norm": area,
                "image_vx_norm_per_sec": vx, "image_vy_norm_per_sec": vy,
                "image_acceleration_norm_per_sec2": acceleration, "image_direction_rad": direction,
                "image_direction_change_rad": direction_change, "log_bbox_expansion_rate_per_sec": expansion,
                "lateral_image_movement_norm_per_sec": vx,
                "kinematics_valid": bool(np.isfinite(vx) and np.isfinite(vy)),
            })
    trajectory = pd.DataFrame(rows, columns=TRACK_FEATURE_FIELDS)
    interactions: list[dict[str, Any]] = []
    for frame_index, frame in trajectory.groupby("frame_index", sort=False):
        vehicles = frame[frame.class_name.isin(VEHICLES)]
        people = frame[frame.class_name == "person"]
        for person in people.itertuples(index=False):
            result: dict[str, Any] = {
                "video_id": video_id, "frame_index": int(frame_index), "timestamp_sec": float(person.timestamp_sec),
                "person_track_id": str(person.track_id), "nearest_vehicle_track_id": None,
                "nearest_vehicle_class": None, "interaction_valid": False, "relative_x_norm": np.nan,
                "relative_y_norm": np.nan, "relative_vx_norm_per_sec": np.nan,
                "relative_vy_norm_per_sec": np.nan, "proximity_norm_diagonal": np.nan,
                "closing_rate_norm_per_sec": np.nan, "time_to_conflict_image_sec": np.nan,
                "within_0_05_norm": False, "within_0_10_norm": False, "within_0_20_norm": False,
            }
            if not vehicles.empty:
                dx = vehicles.center_x_norm.to_numpy(dtype=float) - float(person.center_x_norm)
                dy = vehicles.center_y_norm.to_numpy(dtype=float) - float(person.center_y_norm)
                distances = np.hypot(dx, dy) / math.sqrt(2.0)
                nearest_pos = int(np.argmin(distances))
                vehicle = vehicles.iloc[nearest_pos]
                rel_x, rel_y = float(dx[nearest_pos]), float(dy[nearest_pos])
                rel_vx = float(vehicle.image_vx_norm_per_sec) - float(person.image_vx_norm_per_sec) if np.isfinite(vehicle.image_vx_norm_per_sec) and np.isfinite(person.image_vx_norm_per_sec) else np.nan
                rel_vy = float(vehicle.image_vy_norm_per_sec) - float(person.image_vy_norm_per_sec) if np.isfinite(vehicle.image_vy_norm_per_sec) and np.isfinite(person.image_vy_norm_per_sec) else np.nan
                dist = float(distances[nearest_pos])
                closing = -(rel_x * rel_vx + rel_y * rel_vy) / (math.sqrt(2.0) * max(dist, 1e-12)) if np.isfinite(rel_vx) and np.isfinite(rel_vy) else np.nan
                ttc_proxy = dist / closing if np.isfinite(closing) and closing > 1e-6 else np.nan
                result.update({
                    "nearest_vehicle_track_id": str(vehicle.track_id), "nearest_vehicle_class": str(vehicle.class_name),
                    "interaction_valid": True, "relative_x_norm": rel_x, "relative_y_norm": rel_y,
                    "relative_vx_norm_per_sec": rel_vx, "relative_vy_norm_per_sec": rel_vy,
                    "proximity_norm_diagonal": dist, "closing_rate_norm_per_sec": closing,
                    "time_to_conflict_image_sec": ttc_proxy,
                    "within_0_05_norm": bool(dist <= 0.05), "within_0_10_norm": bool(dist <= 0.10),
                    "within_0_20_norm": bool(dist <= 0.20),
                })
            interactions.append(result)
    return trajectory, pd.DataFrame(interactions, columns=INTERACTION_FIELDS)


def load_completed(status_path: Path) -> dict[str, dict[str, Any]]:
    if not status_path.is_file():
        return {}
    try:
        frame = pd.read_csv(status_path, dtype={"video_id": str})
        return {str(row.video_id): row._asdict() for row in frame.itertuples(index=False)}
    except Exception:
        return {}


def verify_existing(row: dict[str, Any], expected_frames: int, batch_size: int, half_precision: bool) -> bool:
    try:
        saved_batch = int(float(row.get("inference_batch_size", 0)))
        saved_half = str(row.get("inference_half_precision", False)).strip().lower() in {"true", "1"}
        processed_frames = int(float(row.get("processed_frames", -1)))
    except (TypeError, ValueError):
        return False
    if row.get("status") != "completed" or processed_frames != expected_frames or saved_batch != batch_size or saved_half != half_precision:
        return False
    paths = [row.get(key) for key in ("detections_path", "frame_index_path", "tracks_path", "trajectories_path", "interactions_path")]
    return all(isinstance(value, str) and Path(value).is_file() for value in paths)


def process_video(
    model: YOLO, row: Any, expected: dict[str, Any], output_dirs: dict[str, Path],
    tracker_config: Path, device: str, imgsz: int, confidence: float, batch_size: int, half_precision: bool,
    progress_callback: Any = None,
) -> dict[str, Any]:
    video_id = str(row.video_id)
    video_path = Path(row.video_path)
    category = "positive" if int(row.label) == 1 else "negative"
    split = str(row.split)
    start = time.perf_counter()
    inference_seconds = 0.0
    detections: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    processed_frames = 0
    error = ""
    status = "completed"
    fps = float(expected["fps"])
    expected_frames = int(expected["frame_count"])
    width, height = int(expected["width"]), int(expected["height"])
    first_seen: dict[str, float] = {}
    last_seen: dict[str, float] = {}
    try:
        stream = model.track(
            source=str(video_path), stream=True, persist=False, tracker=str(tracker_config),
            imgsz=imgsz, conf=confidence, classes=list(CLASSES), device=device, batch=batch_size,
            quantize=16 if half_precision else 32, verbose=False,
            save=False, show=False,
        )
        for frame_index, result in enumerate(stream):
            processed_frames += 1
            if progress_callback is not None:
                progress_callback(processed_frames)
            inference_seconds += float(result.speed.get("inference", 0.0)) / 1000.0
            timestamp = frame_index / fps
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                frame_rows.append({"video_id": video_id, "frame_index": frame_index, "timestamp_sec": timestamp, "detection_count": 0, "inference_ms": float(result.speed.get("inference", 0.0))})
                continue
            coords = boxes.xyxy.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy().astype(int)
            confs = boxes.conf.detach().cpu().numpy()
            track_ids = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else np.full(len(class_ids), -1, dtype=int)
            frame_rows.append({"video_id": video_id, "frame_index": frame_index, "timestamp_sec": timestamp, "detection_count": len(class_ids), "inference_ms": float(result.speed.get("inference", 0.0))})
            for xyxy, class_id, conf, raw_id in zip(coords, class_ids, confs, track_ids):
                class_id, raw_id = int(class_id), int(raw_id)
                if class_id not in CLASSES:
                    continue
                track_id = f"{class_id}:{raw_id}" if raw_id >= 0 else None
                if track_id is None:
                    age = gap = np.nan
                    track_status = "untracked"
                else:
                    if track_id not in first_seen:
                        first_seen[track_id] = timestamp
                    gap = timestamp - last_seen[track_id] if track_id in last_seen else 0.0
                    age = timestamp - first_seen[track_id]
                    last_seen[track_id] = timestamp
                    track_status = "active"
                detections.append({
                    "video_id": video_id, "frame_index": frame_index, "timestamp_sec": timestamp,
                    "track_id": track_id, "bytetrack_id": raw_id if raw_id >= 0 else None,
                    "class_id": class_id, "class_name": CLASSES[class_id],
                    "x1": float(xyxy[0]), "y1": float(xyxy[1]), "x2": float(xyxy[2]), "y2": float(xyxy[3]),
                    "confidence": float(conf), "track_age_sec": age,
                    "time_since_last_detection_sec": gap, "track_status": track_status,
                })
    except Exception as exc:
        status = "processing_failed"
        error = f"{type(exc).__name__}: {exc}"
    processing_seconds = time.perf_counter() - start
    if status == "completed" and processed_frames != expected_frames:
        status = "frame_count_mismatch"
        error = f"OpenCV/YOLO yielded {processed_frames} frames; integrity scan decoded {expected_frames}"

    det_frame = pd.DataFrame(detections, columns=DET_FIELDS)
    frame_frame = pd.DataFrame(frame_rows, columns=FRAME_FIELDS)
    track_frame = det_frame[det_frame.track_id.notna()].copy() if not det_frame.empty else pd.DataFrame(columns=DET_FIELDS)
    trajectory_frame, interaction_frame = derive_features(video_id, track_frame)
    paths = {
        "detections_path": output_dirs["detections"] / split / f"{video_id}.parquet",
        "frame_index_path": output_dirs["frame_index"] / split / f"{video_id}.parquet",
        "tracks_path": output_dirs["tracks"] / split / f"{video_id}.parquet",
        "trajectories_path": output_dirs["trajectories"] / split / f"{video_id}.parquet",
        "interactions_path": output_dirs["interactions"] / split / f"{video_id}.parquet",
    }
    atomic_parquet(det_frame, paths["detections_path"])
    atomic_parquet(frame_frame, paths["frame_index_path"])
    atomic_parquet(track_frame, paths["tracks_path"])
    atomic_parquet(trajectory_frame, paths["trajectories_path"])
    atomic_parquet(interaction_frame, paths["interactions_path"])
    peak_gpu = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() and device != "cpu" else 0
    peak_gpu_reserved = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() and device != "cpu" else 0
    return {
        "video_id": video_id, "category": category, "label": int(row.label), "split": split,
        "video_path": str(video_path), "status": status, "error": error, "fps": fps,
        "expected_frames": expected_frames, "processed_frames": processed_frames,
        "width": width, "height": height, "duration_seconds": expected_frames / fps if fps > 0 else 0.0,
        "detection_rows": len(det_frame), "tracked_rows": len(track_frame),
        "unique_track_ids": int(track_frame.track_id.nunique()) if not track_frame.empty else 0,
        "trajectory_feature_rows": len(trajectory_frame), "interaction_feature_rows": len(interaction_frame),
        "processing_seconds": round(processing_seconds, 4), "inference_seconds": round(inference_seconds, 4),
        "inference_batch_size": batch_size,
        "inference_half_precision": half_precision,
        "peak_gpu_memory_bytes": peak_gpu, "peak_gpu_reserved_bytes": peak_gpu_reserved,
        **{key: str(path) for key, path in paths.items()},
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }


def validate_outputs(manifest: pd.DataFrame, summary: pd.DataFrame, integrity: pd.DataFrame, root: Path, expected_video_total: int = 1500) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    ids = manifest.video_id.astype(str)
    split_duplicates = int(ids.duplicated().sum())
    checks["manifest_video_count"] = len(manifest)
    checks["unique_manifest_video_ids"] = int(ids.nunique())
    checks["duplicate_manifest_video_ids"] = split_duplicates
    checks["split_counts"] = {s: {"positive": int(((manifest.split == s) & (manifest.label == 1)).sum()),
                                      "negative": int(((manifest.split == s) & (manifest.label == 0)).sum())} for s in SPLITS}
    checks["videos_completed"] = int((summary.status == "completed").sum()) if not summary.empty else 0
    checks["videos_failed"] = int((summary.status != "completed").sum()) if not summary.empty else len(manifest)
    checks["videos_not_processed"] = int(len(set(ids) - set(summary.video_id.astype(str)))) if not summary.empty else len(manifest)
    checks["expected_frame_total"] = int(integrity.frame_count.sum())
    checks["processed_frame_total"] = int(summary.processed_frames.sum()) if not summary.empty else 0
    checks["frame_count_mismatch_videos"] = int((summary.processed_frames != summary.expected_frames).sum()) if not summary.empty else len(manifest)
    checks["duplicate_track_ids_within_frame"] = 0
    checks["invalid_boxes"] = 0
    checks["videos_with_frame_index_gaps"] = 0
    checks["videos_with_detection_count_mismatch"] = 0
    checks["unexpected_nonfinite_trajectory_values"] = 0
    checks["trajectory_rows"] = 0
    checks["interaction_rows"] = 0
    checks["expected_undefined_kinematics"] = {}
    for row in summary.itertuples(index=False):
        if row.status != "completed":
            continue
        frame_index = pd.read_parquet(row.frame_index_path)
        expected_indices = np.arange(int(row.expected_frames), dtype=np.int64)
        if len(frame_index) != int(row.expected_frames) or not np.array_equal(frame_index.frame_index.to_numpy(dtype=np.int64), expected_indices):
            checks["videos_with_frame_index_gaps"] += 1
        detections = pd.read_parquet(row.detections_path)
        per_frame = detections.groupby("frame_index").size()
        expected_counts = frame_index.set_index("frame_index").detection_count
        aligned = per_frame.reindex(expected_counts.index, fill_value=0)
        if not np.array_equal(aligned.to_numpy(dtype=np.int64), expected_counts.to_numpy(dtype=np.int64)):
            checks["videos_with_detection_count_mismatch"] += 1
        invalid_detection = (detections.x1 < 0) | (detections.y1 < 0) | (detections.x2 <= detections.x1) | (detections.y2 <= detections.y1) | (detections.x2 > row.width + 1) | (detections.y2 > row.height + 1)
        checks["invalid_boxes"] += int(invalid_detection.sum())
        tracks = pd.read_parquet(row.tracks_path)
        if not tracks.empty:
            checks["duplicate_track_ids_within_frame"] += int(tracks.duplicated(["frame_index", "track_id"]).sum())
        feats = pd.read_parquet(row.trajectories_path)
        checks["trajectory_rows"] += len(feats)
        for col in ("image_vx_norm_per_sec", "image_vy_norm_per_sec", "image_acceleration_norm_per_sec2", "image_direction_rad", "image_direction_change_rad", "log_bbox_expansion_rate_per_sec", "lateral_image_movement_norm_per_sec"):
            missing = int(feats[col].isna().sum()) if col in feats else 0
            checks["expected_undefined_kinematics"][col] = checks["expected_undefined_kinematics"].get(col, 0) + missing
        numeric = feats.select_dtypes(include=[np.number])
        if not numeric.empty:
            # Derivatives at track starts (or direction/acceleration before enough history) are undefined by definition.
            derivs = [c for c in numeric if c in {"image_vx_norm_per_sec", "image_vy_norm_per_sec", "image_acceleration_norm_per_sec2", "image_direction_rad", "image_direction_change_rad", "log_bbox_expansion_rate_per_sec", "lateral_image_movement_norm_per_sec"}]
            checks["unexpected_nonfinite_trajectory_values"] += int((~np.isfinite(numeric.drop(columns=derivs, errors="ignore").to_numpy(dtype=float))).sum())
        interactions = pd.read_parquet(row.interactions_path)
        checks["interaction_rows"] += len(interactions)
        if not interactions.empty:
            # Missing nearest-vehicle measurements are expected when no vehicle track exists in that frame.
            valid = interactions.interaction_valid
            checks.setdefault("invalid_interaction_rows", 0)
            checks["invalid_interaction_rows"] += int((~valid).sum())
            checks.setdefault("valid_interaction_rows_with_missing_distance", 0)
            checks["valid_interaction_rows_with_missing_distance"] += int((valid & interactions.proximity_norm_diagonal.isna()).sum())
            checks.setdefault("valid_interaction_rows_missing_relative_velocity", 0)
            checks["valid_interaction_rows_missing_relative_velocity"] += int((valid & (interactions.relative_vx_norm_per_sec.isna() | interactions.relative_vy_norm_per_sec.isna())).sum())
            checks.setdefault("valid_interaction_rows_without_positive_closing_rate", 0)
            checks["valid_interaction_rows_without_positive_closing_rate"] += int((valid & (interactions.closing_rate_norm_per_sec.isna() | (interactions.closing_rate_norm_per_sec <= 1e-6))).sum())
    checks["integrity_failed_videos_before_processing"] = int((~integrity.readable.astype(bool)).sum())
    checks["pass"] = bool(
        checks["manifest_video_count"] == expected_video_total and checks["unique_manifest_video_ids"] == expected_video_total
        and checks["duplicate_manifest_video_ids"] == 0 and checks["videos_completed"] == expected_video_total
        and checks["videos_failed"] == 0 and checks["videos_not_processed"] == 0
        and checks["frame_count_mismatch_videos"] == 0 and checks["processed_frame_total"] == checks["expected_frame_total"]
        and checks["videos_with_frame_index_gaps"] == 0 and checks["videos_with_detection_count_mismatch"] == 0
        and checks["duplicate_track_ids_within_frame"] == 0 and checks["invalid_boxes"] == 0
        and checks["unexpected_nonfinite_trajectory_values"] == 0 and checks.get("valid_interaction_rows_with_missing_distance", 0) == 0
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("stage4_two_architecture_comparison/data/manifests/split_manifest.csv"))
    parser.add_argument("--integrity-report", type=Path, default=Path("data/metadata/video_integrity_report.csv"))
    parser.add_argument("--model", type=Path, default=Path("stage4_two_architecture_comparison/models/shared/yolov8n.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16, help="Benchmark-selected consecutive frame batch; ByteTrack still updates in frame order")
    parser.add_argument("--device", default="auto", help="auto, cpu, or CUDA device index such as 0")
    parser.add_argument("--gpu-headroom-mb", type=int, default=800, help="Minimum VRAM to reserve; allocator cap includes extra system/display margin")
    parser.add_argument("--full-precision", action="store_true", help="Disable FP16 inference on CUDA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-positive", type=int, default=0, help="Process a deterministic subset with this many positive videos")
    parser.add_argument("--subset-negative", type=int, default=0, help="Process a deterministic subset with this many negative videos")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke limit; 0 means all 1,500 videos")
    parser.add_argument("--video-ids", default="", help="Optional comma-separated exact IDs for a reproducible batch benchmark")
    parser.add_argument("--no-resume", action="store_true", help="Reprocess videos even if output caches are complete")
    parser.add_argument("--progress-update-frames", type=int, default=30, help="Refresh frame progress/ETA after this many newly processed frames")
    parser.add_argument("--progress-refresh-seconds", type=float, default=1.0, help="Minimum seconds between progress display refreshes")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm bars; per-video log lines remain enabled")
    args = parser.parse_args()
    root = args.project_root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    integrity_path = args.integrity_report if args.integrity_report.is_absolute() else root / args.integrity_report
    model_path = args.model if args.model.is_absolute() else root / args.model
    if not model_path.is_file():
        raise FileNotFoundError(f"Local YOLO checkpoint is missing: {model_path}")
    manifest = make_split_outputs(root, manifest_path, args.seed)
    full_manifest = manifest.copy()
    integrity = pd.read_csv(integrity_path, dtype={"filename": str})
    if len(integrity) != 1500 or not integrity.readable.astype(bool).all():
        raise ValueError("Expected 1,500 readable rows in the video integrity report")
    integrity["video_id"] = integrity.filename.map(lambda value: Path(value).stem)
    if integrity.video_id.duplicated().any():
        raise ValueError("Duplicate video IDs in integrity report")
    manifest = manifest.merge(integrity[["video_id", "fps", "frame_count", "width", "height", "duration_seconds"]], on="video_id", how="left", validate="one_to_one")
    if manifest.fps.isna().any():
        raise ValueError("Integrity metadata missing for some manifest videos")
    subset_mode = args.subset_positive > 0 or args.subset_negative > 0
    if subset_mode:
        if args.subset_positive < 0 or args.subset_negative < 0 or args.video_ids or args.limit:
            raise ValueError("Subset counts must be nonnegative and cannot be combined with --video-ids or --limit")
        subset_name = f"subset_{args.subset_positive}pos_{args.subset_negative}neg"
        manifest = make_stratified_subset(
            manifest, args.subset_positive, args.subset_negative, args.seed,
            root / "data" / "splits" / subset_name,
        )
    else:
        subset_name = ""
    if args.video_ids:
        requested_ids = [v.strip() for v in args.video_ids.split(",") if v.strip()]
        unknown = set(requested_ids) - set(manifest.video_id)
        if unknown:
            raise ValueError(f"Unknown video IDs requested: {sorted(unknown)}")
        manifest = manifest[manifest.video_id.isin(requested_ids)].copy()
        if args.limit > 0:
            raise ValueError("Specify either --video-ids or --limit, not both")
    if args.limit > 0:
        # Preserve both categories in the smoke check.
        chosen = []
        for split in SPLITS:
            for label in (1, 0):
                chosen.extend(manifest[(manifest.split == split) & (manifest.label == label)].head(max(1, args.limit // 6)).index.tolist())
        manifest = manifest.loc[sorted(set(chosen))].copy()

    dirs = {
        "detections": root / "data" / "detections",
        "frame_index": root / "data" / "detections" / "frames",
        "tracks": root / "data" / "tracks",
        "trajectories": root / "data" / "features" / "trajectories",
        "interactions": root / "data" / "features" / "interactions",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    metadata_dir = root / "data" / "metadata"
    report_stem = f"{subset_name}_preprocessing" if subset_mode else "full_preprocessing"
    status_path = metadata_dir / f"{report_stem}_summary.csv"
    failures_path = metadata_dir / f"{report_stem}_failures.csv"
    validation_path = metadata_dir / f"{report_stem}_validation.json"
    report_path = metadata_dir / f"{report_stem}_report.md"
    previous = load_completed(status_path) if not args.no_resume else {}
    records = list(previous.values())
    known = {str(r.get("video_id")): r for r in records}
    if args.device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device '{device}' requested but CUDA is unavailable")
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")
    if args.progress_update_frames < 1 or args.progress_refresh_seconds <= 0:
        raise ValueError("progress update frames and refresh seconds must be positive")
    if args.gpu_headroom_mb < 800:
        raise ValueError("gpu-headroom-mb must be at least 800 MB")
    half_precision = device != "cpu" and not args.full_precision
    gpu_cap_fraction = None
    if device != "cpu":
        device_index = int(device)
        total_gpu_bytes = int(torch.cuda.get_device_properties(device_index).total_memory)
        # Leave the requested reserve plus 200 MiB for display/driver allocations outside PyTorch.
        # Reserve the requested 800 MiB plus an additional 800 MiB for driver/display allocations.
        gpu_cap_fraction = min(0.90, max(0.10, (total_gpu_bytes - (args.gpu_headroom_mb + 800) * 1024**2) / total_gpu_bytes))
        torch.cuda.set_per_process_memory_fraction(gpu_cap_fraction, device=device_index)
    cv2.setNumThreads(1)
    # Variable final-batch shapes make cuDNN autotuning repeatedly benchmark kernels and
    # can dominate inference time; keep it off for steady batched video throughput.
    torch.backends.cudnn.benchmark = False
    model = YOLO(str(model_path))
    tracker_base = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml").read_text(encoding="utf-8"))
    tracker_base["track_buffer"] = 180  # approximately six seconds at the dataset's ~30 FPS
    tracker_config = metadata_dir / "bytetrack_full_frame.yaml"
    tracker_config.write_text(yaml.safe_dump(tracker_base, sort_keys=False), encoding="utf-8")
    run_config = {
        "pipeline": "full-dataset-preprocessing", "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "model": str(model_path), "model_name": "YOLOv8n", "imgsz": args.imgsz,
        "confidence": args.confidence, "classes": CLASSES, "tracker": "ByteTrack", "tracker_config": str(tracker_config),
        "device": device, "batch_size_frames": args.batch_size, "half_precision": half_precision,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "gpu_headroom_target_bytes": args.gpu_headroom_mb * 1024 * 1024, "gpu_process_memory_fraction_cap": gpu_cap_fraction,
        "subset": {"positive": args.subset_positive, "negative": args.subset_negative} if subset_mode else None,
        "sequential_videos": True, "inference_frame_sampling": "every decoded frame",
        "timestamp_method": "frame_index divided by integrity-reported per-video FPS",
        "resume_enabled": not args.no_resume, "limit": args.limit,
        "feature_units": "normalized image coordinates and image-space rates; not physical speed/distance",
    }
    atomic_json(run_config, metadata_dir / f"{report_stem}_config.json")
    total = len(manifest)
    start_all = time.perf_counter()
    expected_total_frames = int(manifest.frame_count.sum())
    progress = tqdm(total=expected_total_frames, desc="Preprocessing frames", unit="frame", dynamic_ncols=True,
                    mininterval=args.progress_refresh_seconds, disable=args.no_progress)
    processed_this_run = 0
    run_detections = 0
    run_tracks = 0
    run_failures = 0
    run_completed_videos = 0
    rows = list(manifest.itertuples(index=False))
    for index, row in enumerate(rows, 1):
        video_id = str(row.video_id)
        exp = {"fps": row.fps, "frame_count": row.frame_count, "width": row.width, "height": row.height}
        existing = known.get(video_id)
        if existing and verify_existing(existing, int(row.frame_count), args.batch_size, half_precision):
            progress.update(int(row.frame_count))
            tqdm.write(f"[{index}/{total}] resume-skip {video_id} ({existing.get('split')})")
            continue
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.reset_peak_memory_stats()
        last_progress = 0

        def update_progress(frame_count: int, force: bool = False) -> None:
            nonlocal last_progress, processed_this_run
            delta = frame_count - last_progress
            expected_for_video = int(row.frame_count)
            if delta > 0 and (force or delta >= args.progress_update_frames or frame_count >= expected_for_video):
                progress.update(delta)
                last_progress = frame_count
                processed_this_run += delta

        try:
            record = process_video(model, row, exp, dirs, tracker_config, device, args.imgsz, args.confidence, args.batch_size, half_precision, update_progress)
        except Exception as exc:
            record = {
                "video_id": video_id, "category": "positive" if int(row.label) else "negative",
                "label": int(row.label), "split": str(row.split), "video_path": str(row.video_path),
                "status": "processing_failed", "error": f"{type(exc).__name__}: {exc}",
                "fps": float(row.fps), "expected_frames": int(row.frame_count), "processed_frames": 0,
                "width": int(row.width), "height": int(row.height), "duration_seconds": float(row.duration_seconds),
                "detection_rows": 0, "tracked_rows": 0, "unique_track_ids": 0,
                "trajectory_feature_rows": 0, "interaction_feature_rows": 0,
                "processing_seconds": 0.0, "inference_seconds": 0.0, "inference_batch_size": args.batch_size,
                "inference_half_precision": half_precision, "peak_gpu_memory_bytes": 0, "peak_gpu_reserved_bytes": 0,
                **{key: "" for key in ("detections_path", "frame_index_path", "tracks_path", "trajectories_path", "interactions_path")},
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
        update_progress(int(record.get("processed_frames", 0)), force=True)
        known[video_id] = record
        run_detections += int(record.get("detection_rows", 0))
        run_tracks += int(record.get("tracked_rows", 0))
        run_failures += int(record.get("status") != "completed")
        run_completed_videos += int(record.get("status") == "completed")
        ordered = pd.DataFrame([known[k] for k in sorted(known)], columns=SUMMARY_FIELDS)
        write_csv_atomic(ordered, status_path)
        elapsed = max(time.perf_counter() - start_all, 1e-9)
        progress.set_postfix({"video": f"{index}/{total}", "fps": f"{processed_this_run / elapsed:.1f}",
                              "det": run_detections, "tracks": run_tracks, "failed": run_failures}, refresh=True)
        tqdm.write(f"[{index}/{total}] {video_id} {record['status']} frames={record['processed_frames']}/{record['expected_frames']} det={record['detection_rows']} tracks={record['tracked_rows']} seconds={record['processing_seconds']:.1f}")
    progress.close()

    summary = pd.DataFrame([known[k] for k in sorted(known)], columns=SUMMARY_FIELDS)
    # For smoke runs, validate only selected rows while clearly preserving the requested all-dataset target.
    checks = validate_outputs(manifest, summary[summary.video_id.isin(manifest.video_id)], integrity[integrity.video_id.isin(manifest.video_id)], root, expected_video_total=len(manifest))
    checks["requested_total_videos"] = len(manifest)
    checks["full_dataset_video_total"] = len(full_manifest)
    checks["this_invocation_videos"] = total
    checks["elapsed_seconds_this_invocation"] = round(time.perf_counter() - start_all, 3)
    elapsed_for_metrics = max(float(checks["elapsed_seconds_this_invocation"]), 1e-9)
    checks["performance_metrics"] = {
        "completed_videos_per_hour_this_invocation": round(run_completed_videos / elapsed_for_metrics * 3600, 3),
        "processed_frames_per_second_this_invocation": round(processed_this_run / elapsed_for_metrics, 3),
        "yolo_inference_frames_per_second_completed_scope": round(checks["processed_frame_total"] / max(float(summary.loc[summary.status == "completed", "inference_seconds"].sum()), 1e-9), 3) if checks["videos_completed"] else None,
        "mean_yolo_inference_ms_per_frame": round(float(summary.loc[summary.status == "completed", "inference_seconds"].sum()) * 1000 / checks["processed_frame_total"], 3) if checks["processed_frame_total"] else None,
        "per_video_pipeline_frames_per_second_completed_scope": round(checks["processed_frame_total"] / max(float(summary.loc[summary.status == "completed", "processing_seconds"].sum()), 1e-9), 3) if checks["videos_completed"] else None,
        "mean_processing_seconds_per_completed_video": round(float(summary.loc[summary.status == "completed", "processing_seconds"].mean()), 3) if checks["videos_completed"] else None,
        "detections_per_processed_frame": round(float(summary.detection_rows.sum()) / checks["processed_frame_total"], 5) if checks["processed_frame_total"] else None,
        "mean_gpu_allocated_mib_per_video_peak": round(float(summary.loc[summary.status == "completed", "peak_gpu_memory_bytes"].mean()) / (1024 ** 2), 2) if checks["videos_completed"] else None,
        "peak_gpu_allocated_mib": round(float(summary.peak_gpu_memory_bytes.max()) / (1024 ** 2), 2) if not summary.empty else 0.0,
        "peak_gpu_reserved_mib": round(float(summary.peak_gpu_reserved_bytes.max()) / (1024 ** 2), 2) if not summary.empty else 0.0,
        "yolov8_checkpoint_size_mib": round(model_path.stat().st_size / (1024 ** 2), 3),
        "failed_video_count": int(checks["videos_failed"]),
        "note": "Preprocessing throughput/resource metrics only; these are not detector accuracy, precision, recall, F1, or collision prediction metrics.",
    }
    checks["feature_semantics"] = {
        "image_velocity_acceleration_direction_and_box_expansion": "computed from consecutive tracked observations using elapsed timestamps",
        "relative_vehicle_person_features": "nearest tracked vehicle to each tracked person in the same frame",
        "time_to_conflict_image_sec": "normalized image-space separation divided by positive normalized image-space closing rate; not physical TTC",
        "undefined_derivatives": "left null for observations without enough prior track history; no imputation",
        "interaction_without_vehicle": "interaction_valid=false and nearest-vehicle metrics are not applicable",
    }
    atomic_json(checks, validation_path)
    failed = summary[summary.status != "completed"] if not summary.empty else pd.DataFrame(columns=SUMMARY_FIELDS)
    write_csv_atomic(failed, failures_path)
    report = [
        "# Full Dataset Preprocessing Report" if not subset_mode else "# Stratified Subset Preprocessing Report", "", f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "", "## Scope", "", "YOLOv8n and ByteTrack were run sequentially over every decoded frame in the selected manifest, followed by image-space trajectory and nearest vehicle/person interaction features. No model fitting, feature selection, video download, or source-video modification was performed.",
        "", "## Counts", "", f"- Videos represented in current manifest: {len(manifest)}", f"- Completed videos in saved summary: {checks['videos_completed']}", f"- Failed/incomplete videos: {checks['videos_failed']}", f"- Frames expected: {checks['expected_frame_total']}", f"- Frames processed: {checks['processed_frame_total']}", f"- Detection rows: {int(summary.detection_rows.sum()) if not summary.empty else 0}", f"- Track rows: {int(summary.tracked_rows.sum()) if not summary.empty else 0}", f"- Trajectory feature rows: {checks['trajectory_rows']}", f"- Interaction feature rows: {checks['interaction_rows']}",
        "", "## Performance", "", f"- YOLO inference throughput: {checks['performance_metrics']['yolo_inference_frames_per_second_completed_scope']:.2f} frames/sec" if checks['performance_metrics']['yolo_inference_frames_per_second_completed_scope'] is not None else "- YOLO inference throughput: unavailable (no completed clips)", f"- Per-video pipeline throughput (decode through Parquet write): {checks['performance_metrics']['per_video_pipeline_frames_per_second_completed_scope']:.2f} frames/sec" if checks['performance_metrics']['per_video_pipeline_frames_per_second_completed_scope'] is not None else "- Per-video pipeline throughput: unavailable (no completed clips)", f"- Mean YOLO inference latency: {checks['performance_metrics']['mean_yolo_inference_ms_per_frame']} ms/frame", f"- Mean completed video time: {checks['performance_metrics']['mean_processing_seconds_per_completed_video']} seconds", f"- Peak PyTorch CUDA allocation: {checks['performance_metrics']['peak_gpu_allocated_mib']} MiB", f"- Peak PyTorch CUDA reserved memory: {checks['performance_metrics']['peak_gpu_reserved_mib']} MiB", f"- YOLOv8n checkpoint file size: {checks['performance_metrics']['yolov8_checkpoint_size_mib']} MiB", "- These are preprocessing/resource metrics; no accuracy or F1 evaluation was run.",
        "", "## Outputs", "", "- Splits: `data/splits/`", "- Per-object detections: `data/detections/{train,validation,test}/`", "- Dense per-frame index, including frames with zero detections: `data/detections/frames/{train,validation,test}/`", "- Tracks: `data/tracks/{train,validation,test}/`", "- Trajectory features: `data/features/trajectories/{train,validation,test}/`", "- Interaction features: `data/features/interactions/{train,validation,test}/`", f"- Per-video summary: `data/metadata/{report_stem}_summary.csv`", f"- Failures: `data/metadata/{report_stem}_failures.csv`", f"- Validation: `data/metadata/{report_stem}_validation.json`", f"- Configuration: `data/metadata/{report_stem}_config.json`", "", "## Interpretation", "", "Trajectory rates, proximity, closing rate, and time-to-conflict are image-space proxies. They are not physical speed, distance, collision probability, or calibrated TTC. Null motion values at track starts are mathematically undefined; interaction metrics are not applicable when no vehicle track is present. See the machine-readable validation JSON for exact checks and counts.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"validation_pass": checks["pass"], "videos_completed": checks["videos_completed"], "videos_failed": checks["videos_failed"], "frames_expected": checks["expected_frame_total"], "frames_processed": checks["processed_frame_total"], "performance_metrics": checks["performance_metrics"], "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
