"""Build ego-corridor conflict features from the shared perception cache."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd() / "stage4_two_architecture_comparison"))
from src.architecture_2.features import make_horizon_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perception-run", type=Path, required=True)
    parser.add_argument("--camera-run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("stage4_two_architecture_comparison/data/manifests/split_manifest.csv"))
    args = parser.parse_args()
    root = Path.cwd()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    perception = args.perception_run if args.perception_run.is_absolute() else root / args.perception_run
    camera_run = args.camera_run if args.camera_run.is_absolute() else root / args.camera_run
    manifest = pd.read_csv(manifest_path, dtype={"video_id": str})
    features = make_horizon_examples(manifest, perception / "tracks", camera_run / "camera_motion")
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_architecture2_features"
    run_dir = root / "stage4_two_architecture_comparison" / "experiments" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "run_id": run_id,
        "architecture": "Architecture 2 feature generation",
        "dataset_subset": "400-video preprocessing subset",
        "manifest": str(manifest_path),
        "perception_run": str(perception),
        "camera_motion_run": str(camera_run),
        "horizons_sec": [0.5, 1.0, 1.5, 2.0, 4.0],
        "observation_sec": 3.0,
        "geometry": {"mode": "image_space_heuristic", "metric_geometry_valid": False},
        "target_rule": "event occurs in (endpoint, endpoint+horizon]; post-event frames excluded",
        "scope": "400-video subset; no-camera analysis",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    features.to_parquet(run_dir / "features.parquet", index=False)
    features.to_csv(run_dir / "features.csv", index=False)
    with (run_dir / "experiment_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["run_id", "architecture", "model", "dataset_subset", "split", "configuration", "seed", "completion_status", "checkpoint_location", "report_location"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"run_id": run_id, "architecture": "Architecture 2 feature generation", "model": "image-space ego corridor and risk features", "dataset_subset": "400-video preprocessing subset", "split": "400-video subset; inherited video-level split", "configuration": str(run_dir / "config.json"), "seed": 42, "completion_status": "completed", "checkpoint_location": "not applicable", "report_location": str(run_dir / "features.parquet")})
    print(f"run_id={run_id} rows={len(features)} path={run_dir}")
    print(features.groupby(["horizon_sec", "split", "label"]).size().to_string())


if __name__ == "__main__":
    main()

