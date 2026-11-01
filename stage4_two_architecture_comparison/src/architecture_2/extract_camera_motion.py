"""Extract sparse background affine motion using cached detections as masks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from camera_motion import estimate_camera_motion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("stage4_two_architecture_comparison/data/manifests/split_manifest.csv"))
    parser.add_argument("--perception-run", type=Path, required=True)
    parser.add_argument("--sample-rate", type=float, default=5.0)
    parser.add_argument("--limit-per-class", type=int, default=0)
    args = parser.parse_args()
    root = Path.cwd()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    perception = args.perception_run if args.perception_run.is_absolute() else root / args.perception_run
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rows = [r for r in rows if Path(r["video_path"]).is_file()]
    if args.limit_per_class:
        rows = [r for lab in ("0", "1") for r in [x for x in rows if x["label"] == lab][:args.limit_per_class]]
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_architecture2_camera_motion"
    run_dir = root / "stage4_two_architecture_comparison" / "experiments" / "runs" / run_id
    out_dir = run_dir / "camera_motion"
    out_dir.mkdir(parents=True, exist_ok=False)
    config = {"run_id": run_id, "architecture": "Architecture 2 camera motion", "perception_run": str(perception), "sample_rate_fps": args.sample_rate, "method": "masked sparse optical flow + RANSAC partial affine", "units": "image pixels per sampled frame", "metric_geometry": False, "videos": [r["video_id"] for r in rows]}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    summary = []
    for row in rows:
        vid = str(row["video_id"])
        cap = cv2.VideoCapture(row["video_path"])
        if not cap.isOpened():
            summary.append({"video_id": vid, "status": "open_failed"})
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        det_path = perception / "detections" / f"{vid}.parquet"
        detections = pd.read_parquet(det_path)
        boxes_by_frame = {int(k): g[["x1", "y1", "x2", "y2"]].to_numpy().tolist() for k, g in detections.groupby("frame_index")}
        next_sample = 0.0
        interval = 1.0 / args.sample_rate
        index = -1
        previous_gray = None
        previous_boxes = []
        output = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not math.isfinite(timestamp) or (index > 0 and timestamp <= 0):
                timestamp = index / fps
            if timestamp + 1e-6 < next_sample:
                continue
            next_sample += interval
            while next_sample <= timestamp + 1e-6:
                next_sample += interval
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion = estimate_camera_motion(previous_gray, gray, previous_boxes) if previous_gray is not None else {"valid": False, "dx_px": 0.0, "dy_px": 0.0, "scale": 1.0, "rotation_rad": 0.0, "inliers": 0, "inlier_ratio": 0.0, "uncertainty": 1.0}
            output.append({"video_id": vid, "frame_index": index, "timestamp_sec": float(timestamp), **motion})
            previous_gray = gray
            previous_boxes = boxes_by_frame.get(index, [])
        cap.release()
        pd.DataFrame(output).to_parquet(out_dir / f"{vid}.parquet", index=False)
        valid = [r for r in output if r["valid"]]
        summary.append({"video_id": vid, "status": "ok", "sampled_frames": len(output), "valid_estimates": len(valid), "valid_fraction": len(valid) / max(len(output), 1), "median_inlier_ratio": float(pd.DataFrame(valid).inlier_ratio.median()) if valid else 0.0})
    pd.DataFrame(summary).to_csv(run_dir / "video_summary.csv", index=False)
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir), "videos": len(summary), "successful": sum(r.get("status") == "ok" for r in summary)}, indent=2))


if __name__ == "__main__":
    main()
