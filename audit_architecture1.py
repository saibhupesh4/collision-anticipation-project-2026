import pandas as pd
from pathlib import Path

manifest_path = Path(
    "stage4_two_architecture_comparison/data/manifests/subset_400_manifest.csv"
)

run_dir = Path(
    "stage4_two_architecture_comparison/experiments/runs/"
    "run_20261101_102221_architecture1_features"
)

manifest = pd.read_csv(manifest_path, dtype={"video_id": str})
features = pd.read_parquet(run_dir / "features.parquet")

horizons = [0.5, 1.0, 1.5, 2.0, 4.0]

positive_videos = set(
    manifest.loc[manifest["label"] == 1, "video_id"]
)

for horizon in horizons:
    actual = set(
        features.loc[
            (features["horizon_sec"] == horizon) &
            (features["label"] == 1),
            "video_id"
        ]
    )

    missing = sorted(positive_videos - actual)

    print(f"\nHorizon: {horizon}s")
    print(f"Expected positive videos: {len(positive_videos)}")
    print(f"Generated positive examples: {len(actual)}")
    print(f"Missing video IDs: {missing}")
