"""Run one shared YOLOv8n + ByteTrack pass over manifest videos."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import torch
import yaml
from ultralytics import YOLO
import ultralytics


CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FIELDS = [
    "video_id", "frame_index", "timestamp_sec", "track_id", "bytetrack_id", "class_id", "class_name",
    "x1", "y1", "x2", "y2", "confidence", "track_age_sec", "time_since_last_detection_sec",
    "track_status",
]


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("stage4_two_architecture_comparison/data/manifests/split_manifest.csv"))
    parser.add_argument("--sample-rate", type=float, default=5.0)
    parser.add_argument("--limit-per-class", type=int, default=0, help="0 processes all manifest videos")
    parser.add_argument("--device", default="0")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sample_rate <= 0 or args.imgsz <= 0:
        raise ValueError("sample-rate and imgsz must be positive")

    root = Path.cwd().resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else (root / args.manifest)
    rows = [r for r in load_manifest(manifest_path) if Path(r["video_path"]).is_file()]
    if args.limit_per_class:
        selected = []
        for label in ("0", "1"):
            selected.extend([r for r in rows if r["label"] == label][:args.limit_per_class])
        rows = selected
    if not rows:
        raise FileNotFoundError("No manifest video files are available locally")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{now}_shared_{args.sample_rate:g}fps"
    run_dir = root / "stage4_two_architecture_comparison" / "experiments" / "runs" / run_id
    det_dir, track_dir = run_dir / "detections", run_dir / "tracks"
    det_dir.mkdir(parents=True, exist_ok=False)
    track_dir.mkdir(parents=True, exist_ok=False)

    default_tracker = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
    tracker_params = yaml.safe_load(default_tracker.read_text(encoding="utf-8"))
    tracker_params["track_buffer"] = int(round(6.0 * args.sample_rate))
    tracker_file = run_dir / "bytetrack.yaml"
    tracker_file.write_text(yaml.safe_dump(tracker_params, sort_keys=False), encoding="utf-8")
    config = {
        "run_id": run_id,
        "architecture": "shared perception",
        "model": args.model,
        "imgsz": args.imgsz,
        "confidence": args.confidence,
        "classes": CLASSES,
        "tracker": "ByteTrack",
        "tracker_config": str(tracker_file),
        "tracker_parameters": tracker_params,
        "lost_track_buffer_seconds": 6.0,
        "sample_rate_fps": args.sample_rate,
        "sampling_rule": "sample the first decoded frame at or after each clip-relative 1/fps time-grid point",
        "device": args.device,
        "half_precision": False,
        "seed": args.seed,
        "manifest": str(manifest_path),
        "videos": [r["video_id"] for r in rows],
        "smoke_subset_only": len(rows) <= 80,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    try:
        model = YOLO(args.model)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}\n"
        (run_dir / "error.log").write_text(message, encoding="utf-8")
        with (run_dir / "experiment_registry.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run_id", "architecture", "model", "dataset_subset", "split", "configuration", "seed", "completion_status", "failure", "checkpoint_location", "report_location"])
            writer.writeheader()
            writer.writerow({
                "run_id": run_id, "architecture": "shared perception", "model": args.model,
                "dataset_subset": "local smoke subset", "split": "smoke_test_only",
                "configuration": str(run_dir / "config.json"), "seed": args.seed,
                "completion_status": "failed", "failure": message.strip(),
                "checkpoint_location": args.model, "report_location": str(run_dir / "error.log"),
            })
        raise
    if torch.cuda.is_available() and str(args.device) != "cpu":
        torch.cuda.reset_peak_memory_stats()
    config["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() and str(args.device) != "cpu" else "CPU"
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = []
    class_counts: Counter[str] = Counter()
    for row in rows:
        last_seen_by_id.clear()
        video_id = Path(row["video_id"]).stem
        video_path = Path(row["video_path"])
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            summary.append({"video_id": video_id, "label": row["label"], "status": "open_failed"})
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            cap.release()
            summary.append({"video_id": video_id, "label": row["label"], "status": "invalid_fps"})
            continue

        detections: list[dict[str, object]] = []
        track_state: dict[tuple[int, int], tuple[float, int]] = {}
        next_sample = 0.0
        frame_index = -1
        sampled = 0
        elapsed_inference = 0.0
        video_start = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not math.isfinite(timestamp) or (frame_index > 0 and timestamp <= 0):
                timestamp = frame_index / fps
            if timestamp + 1e-6 < next_sample:
                continue
            interval = 1.0 / args.sample_rate
            next_sample += interval
            while next_sample <= timestamp + 1e-6:
                next_sample += interval
            sampled += 1
            infer_start = time.perf_counter()
            results = model.track(
                frame,
                persist=True,
                tracker=str(tracker_file),
                imgsz=args.imgsz,
                conf=args.confidence,
                classes=list(CLASSES),
                device=args.device,
                verbose=False,
            )
            elapsed_inference += time.perf_counter() - infer_start
            result = results[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            cls = boxes.cls.detach().cpu().numpy().astype(int)
            conf = boxes.conf.detach().cpu().numpy()
            ids = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else [-1] * len(cls)
            for coords, class_id, confidence, track_id in zip(xyxy, cls, conf, ids):
                if class_id not in CLASSES:
                    continue
                if track_id >= 0:
                    # ByteTrack itself is class-agnostic. Keep its raw ID for
                    # audit, but namespace the exported ID by class so a class
                    # flicker cannot merge unlike objects in downstream models.
                    raw_track_id = int(track_id)
                    scoped_key = (int(class_id), raw_track_id)
                    scoped_track_id = f"{int(class_id)}:{raw_track_id}"
                    previous = track_state.get(scoped_key)
                    if previous is None:
                        track_state[scoped_key] = (timestamp, 1)
                        gap = 0.0
                        track_age = 0.0
                    else:
                        first_seen_saved, age_samples_saved = previous
                        last_seen = last_seen_by_id.get(scoped_key, timestamp)
                        gap = max(0.0, timestamp - last_seen)
                        track_age = max(0.0, timestamp - first_seen_saved)
                        track_state[scoped_key] = (first_seen_saved, age_samples_saved + 1)
                    last_seen_by_id[scoped_key] = timestamp
                    age = track_age
                    since_last = gap
                    status = "active"
                else:
                    raw_track_id, scoped_track_id = None, None
                    age, since_last, status = 0.0, 0.0, "untracked"
                detections.append({
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "timestamp_sec": round(float(timestamp), 6),
                    "track_id": scoped_track_id,
                    "bytetrack_id": raw_track_id,
                    "class_id": int(class_id),
                    "class_name": CLASSES[int(class_id)],
                    "x1": float(coords[0]), "y1": float(coords[1]),
                    "x2": float(coords[2]), "y2": float(coords[3]),
                    "confidence": float(confidence),
                    "track_age_sec": float(age),
                    "time_since_last_detection_sec": float(since_last),
                    "track_status": status,
                })
                class_counts[CLASSES[int(class_id)]] += 1

        cap.release()
        # The cache intentionally stores the same shared output in separate detection
        # and track tables so both architectures consume identical perception results.
        clean = pd.DataFrame(detections, columns=FIELDS)
        det_path, track_path = det_dir / f"{video_id}.parquet", track_dir / f"{video_id}.parquet"
        clean.to_parquet(det_path, index=False)
        clean[clean["track_id"].notna()].to_parquet(track_path, index=False)
        if hasattr(model, "predictor") and model.predictor is not None:
            for tracker in getattr(model.predictor, "trackers", []):
                tracker.reset()
        video_elapsed = time.perf_counter() - video_start
        summary.append({
            "video_id": video_id,
            "label": row["label"],
            "status": "ok",
            "fps_source": round(float(fps), 4),
            "duration_sec": round(float(frame_index / fps), 4),
            "sampled_frames": sampled,
            "effective_sample_fps": round(sampled / max(frame_index / fps, 1e-6), 4),
            "detection_rows": len(clean),
            "tracked_rows": int(clean["track_id"].notna().sum()),
            "unique_tracks": int(clean.loc[clean["track_id"].notna(), "track_id"].nunique()),
            "inference_seconds": round(elapsed_inference, 4),
            "processing_seconds": round(video_elapsed, 4),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() and str(args.device) != "cpu" else 0,
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() and str(args.device) != "cpu" else 0,
            "detections_path": str(det_path),
            "tracks_path": str(track_path),
        })

    pd.DataFrame(summary).to_csv(run_dir / "video_summary.csv", index=False)
    (run_dir / "class_detection_counts.json").write_text(json.dumps(class_counts, indent=2), encoding="utf-8")
    with (run_dir / "experiment_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "architecture", "model", "dataset_subset", "split", "configuration", "seed", "completion_status", "checkpoint_location", "report_location"])
        writer.writeheader()
        writer.writerow({
            "run_id": run_id, "architecture": "shared perception", "model": args.model,
            "dataset_subset": "local smoke subset" if len(rows) <= 80 else "manifest subset",
            "split": "smoke_test_only", "configuration": str(run_dir / "config.json"),
            "seed": args.seed, "completion_status": "completed_with_failures" if any(r.get("status") != "ok" for r in summary) else "completed",
            "checkpoint_location": args.model, "report_location": str(run_dir / "video_summary.csv"),
        })
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "videos": len(summary), "successful": sum(r.get("status") == "ok" for r in summary), "classes": class_counts}, indent=2))


if __name__ == "__main__":
    last_seen_by_id: dict[int, float] = {}
    main()
