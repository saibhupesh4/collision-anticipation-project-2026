import pandas as pd
import numpy as np
from pathlib import Path

run_dir = Path(
    "stage4_two_architecture_comparison/experiments/runs/"
    "run_20261101_102221_architecture1_features"
)

features_path = run_dir / "features.parquet"
sequence_path = run_dir / "sequence_metadata.csv"

df = pd.read_parquet(features_path)

print("=" * 60)
print("ARCHITECTURE 1 FEATURE VALIDATION")
print("=" * 60)

# 1. Dataset dimensions
print("\n1. DATASET SIZE")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Columns:", df.columns.tolist())

# 2. Missing values
print("\n2. MISSING VALUES")
missing = df.isnull().sum()
missing = missing[missing > 0]

if missing.empty:
    print("No missing values.")
else:
    print(missing.to_string())

# 3. Infinite numeric values
print("\n3. INFINITE VALUES")
numeric = df.select_dtypes(include=[np.number])
inf_counts = pd.Series(
    np.isinf(numeric.to_numpy()).sum(axis=0),
    index=numeric.columns
)
inf_counts = inf_counts[inf_counts > 0]

if inf_counts.empty:
    print("No infinite values.")
else:
    print(inf_counts.to_string())

# 4. Label distribution
print("\n4. LABEL DISTRIBUTION")
print(
    df.groupby(["horizon_sec", "split", "label"])
      .size()
      .to_string()
)

# 5. Duplicate examples
print("\n5. DUPLICATE CHECK")
key_cols = [
    c for c in ["video_id", "horizon_sec", "label", "split"]
    if c in df.columns
]

if key_cols:
    duplicates = df.duplicated(subset=key_cols).sum()
    print("Duplicate rows using", key_cols, ":", duplicates)
else:
    print("Could not identify duplicate-check columns.")

# 6. Video split consistency
print("\n6. VIDEO SPLIT CONSISTENCY")
if {"video_id", "split"}.issubset(df.columns):
    split_counts = df.groupby("video_id")["split"].nunique()
    inconsistent = split_counts[split_counts > 1]
    print("Unique videos:", df["video_id"].nunique())
    print("Videos assigned to multiple splits:", len(inconsistent))
    if len(inconsistent):
        print(inconsistent.to_string())
else:
    print("Required columns unavailable.")

# 7. Timestamp-related columns
print("\n7. TIMESTAMP AUDIT")
time_cols = [
    c for c in df.columns
    if any(term in c.lower() for term in
           ["timestamp", "endpoint", "event_time", "observation_time"])
]
print("Timestamp-related columns:", time_cols)

if time_cols:
    print(df[time_cols].describe(include="all").to_string())
else:
    print("No explicit timestamp columns found in feature table.")

# 8. GRU sequence metadata
print("\n8. GRU SEQUENCE METADATA")
if sequence_path.exists():
    seq = pd.read_csv(sequence_path, dtype={"video_id": str})
    print("Sequence metadata rows:", len(seq))
    print("Sequence columns:", seq.columns.tolist())
    if {"horizon_sec", "split", "label"}.issubset(seq.columns):
        print(
            seq.groupby(["horizon_sec", "split", "label"])
               .size().to_string()
        )
else:
    print("Sequence metadata file not found.")

print("\n" + "=" * 60)
print("AUDIT COMPLETE — NO FILES MODIFIED")
print("=" * 60)
