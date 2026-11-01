import pandas as pd
from pathlib import Path

manifest = pd.read_csv(
    "stage4_two_architecture_comparison/data/manifests/subset_400_manifest.csv",
    dtype={"video_id": str}
)

row = manifest[manifest["video_id"] == "00332"]
print("Manifest record:")
print(row.to_string(index=False))

track_path = Path("data/cache_flat_400/tracks/00332.parquet")
tracks = pd.read_parquet(track_path)

print("\nTrack columns:")
print(tracks.columns.tolist())

print("\nTrack time range:")
for col in ["timestamp_sec", "time_sec", "frame_time_sec", "time"]:
    if col in tracks.columns:
        print(col, tracks[col].min(), tracks[col].max())
