"""Build event-relative static and trajectory features from shared tracks."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from features import GRU_FEATURES, make_horizon_examples, make_temporal_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("stage4_two_architecture_comparison/data/manifests/split_manifest.csv"))
    parser.add_argument("--tracks-dir", type=Path, required=True)
    parser.add_argument("--observation-sec", type=float, default=3.0)
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else Path.cwd() / args.manifest
    tracks_dir = args.tracks_dir if args.tracks_dir.is_absolute() else Path.cwd() / args.tracks_dir
    manifest = pd.read_csv(manifest_path, dtype={"video_id": str})
    examples = make_horizon_examples(manifest, tracks_dir, observation_sec=args.observation_sec)
    sequence_metadata, sequences = make_temporal_examples(manifest, tracks_dir)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_architecture1_features"
    out_dir = Path.cwd() / "stage4_two_architecture_comparison" / "experiments" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "run_id": run_id,
        "architecture": "Architecture 1 feature generation",
        "dataset_subset": "400-video preprocessing subset",
        "manifest": str(manifest_path),
        "track_cache": str(tracks_dir),
        "observation_sec": args.observation_sec,
        "horizons_sec": [0.5, 1.0, 1.5, 2.0, 4.0],
        "seed": 42,
        "target_rule": "event occurs in (endpoint, endpoint+horizon]; post-event frames excluded",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    examples.to_parquet(out_dir / "features.parquet", index=False)
    examples.to_csv(out_dir / "features.csv", index=False)
    sequence_metadata.to_csv(out_dir / "sequence_metadata.csv", index=False)
    np.savez_compressed(
        out_dir / "sequences.npz",
        sequences=sequences,
        video_id=sequence_metadata.video_id.to_numpy(dtype="U"),
        label=sequence_metadata.label.to_numpy(dtype=np.int64),
        split=sequence_metadata.split.to_numpy(dtype="U"),
        horizon_sec=sequence_metadata.horizon_sec.to_numpy(dtype=np.float32),
        endpoint_sec=sequence_metadata.endpoint_sec.to_numpy(dtype=np.float32),
    )
    (out_dir / "gru_feature_names.json").write_text(json.dumps(list(GRU_FEATURES), indent=2), encoding="utf-8")
    (out_dir / "feature_config.txt").write_text(
        f"track_cache={tracks_dir}\nobservation_sec={args.observation_sec}\nhorizons_sec=0.5,1,1.5,2,4\n"
        "label=event occurs in (endpoint, endpoint+horizon]; post-event frames excluded\n"
        "absolute image-center shortcut features are excluded\n",
        encoding="utf-8",
    )
    with (out_dir / "experiment_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["run_id", "architecture", "model", "dataset_subset", "split", "configuration", "seed", "completion_status", "checkpoint_location", "report_location"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"run_id": run_id, "architecture": "Architecture 1 feature generation", "model": "static and trajectory features + GRU sequences", "dataset_subset": "400-video preprocessing subset", "split": "400-video subset; inherited video-level split", "configuration": str(out_dir / "config.json"), "seed": 42, "completion_status": "completed", "checkpoint_location": "not applicable", "report_location": str(out_dir / "features.parquet")})
    print(f"run_id={run_id} rows={len(examples)} sequence_shape={sequences.shape} path={out_dir}")
    if not examples.empty:
        print(examples.groupby(["horizon_sec", "split", "label"]).size().to_string())


if __name__ == "__main__":
    main()

