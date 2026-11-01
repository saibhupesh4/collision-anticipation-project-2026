"""Export ByteTrack IDs scoped by class without rerunning the detector."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_run.resolve()
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_shared_{config['sample_rate_fps']:g}fps_class_scoped"
    output = Path.cwd() / "stage4_two_architecture_comparison" / "experiments" / "runs" / run_id
    det_out, track_out = output / "detections", output / "tracks"
    det_out.mkdir(parents=True, exist_ok=False)
    track_out.mkdir(parents=True, exist_ok=False)
    config.update({"run_id": run_id, "postprocessed_from_run": source.name, "track_id_namespace": "class_id:bytetrack_id", "detection_model_rerun": False})
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    summary = pd.read_csv(source / "video_summary.csv", dtype={"video_id": str}).to_dict("records")
    for row in summary:
        video_id = str(row["video_id"])
        detections = pd.read_parquet(source / "detections" / f"{video_id}.parquet")
        if "bytetrack_id" not in detections:
            detections["bytetrack_id"] = detections["track_id"]
        ids: list[object] = []
        for class_id, raw_id in zip(detections["class_id"], detections["bytetrack_id"]):
            ids.append(f"{int(class_id)}:{int(raw_id)}" if pd.notna(raw_id) and int(raw_id) >= 0 else None)
        detections["track_id"] = ids
        tracked = detections[detections["track_id"].notna()].copy()
        if not tracked.empty:
            tracked.sort_values(["track_id", "timestamp_sec", "frame_index"], inplace=True)
            tracked["track_age_sec"] = tracked.groupby("track_id")["timestamp_sec"].transform(lambda s: s - s.iloc[0])
            tracked["time_since_last_detection_sec"] = tracked.groupby("track_id")["timestamp_sec"].diff().fillna(0.0).clip(lower=0.0)
            tracked["track_status"] = "active"
            age_map = tracked.set_index(["frame_index", "track_id"])[["track_age_sec", "time_since_last_detection_sec", "track_status"]]
            detections.set_index(["frame_index", "track_id"], inplace=True)
            detections.update(age_map)
            detections.reset_index(inplace=True)
        detections.to_parquet(det_out / f"{video_id}.parquet", index=False)
        tracked.sort_values(["frame_index", "track_id"], inplace=True)
        tracked.to_parquet(track_out / f"{video_id}.parquet", index=False)
        row["tracked_rows"] = len(tracked)
        row["unique_tracks"] = int(tracked["track_id"].nunique()) if not tracked.empty else 0
        row["detections_path"] = str(det_out / f"{video_id}.parquet")
        row["tracks_path"] = str(track_out / f"{video_id}.parquet")

    pd.DataFrame(summary).to_csv(output / "video_summary.csv", index=False)
    (output / "class_detection_counts.json").write_text((source / "class_detection_counts.json").read_text(encoding="utf-8"), encoding="utf-8")
    with (output / "experiment_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["run_id", "architecture", "model", "dataset_subset", "split", "configuration", "seed", "completion_status", "checkpoint_location", "report_location"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"run_id": run_id, "architecture": "shared perception (class-scoped track IDs)", "model": config["model"], "dataset_subset": "80-video smoke subset", "split": "smoke_test_only", "configuration": str(output / "config.json"), "seed": config.get("seed", 42), "completion_status": "completed", "checkpoint_location": config["model"], "report_location": str(output / "video_summary.csv")})
    print(json.dumps({"run_id": run_id, "source_run": source.name, "videos": len(summary), "run_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
