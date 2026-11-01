"""Build leakage-safe metadata and video-level split manifests for Nexar train."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def optional_float(value: str | None) -> float | str:
    if value is None or not value.strip():
        return ""
    return float(value)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.project_root.resolve()

    source = root / "data" / "metadata"
    output = root / "stage4_two_architecture_comparison" / "data" / "manifests"
    rows: list[dict[str, object]] = []
    for file_name, label in (("positive_metadata.csv", 1), ("negative_metadata.csv", 0)):
        for record in read_rows(source / file_name):
            video_id = Path(record["file_name"]).stem
            category = "positive" if label else "negative"
            video_path = root / "data" / "raw" / category / record["file_name"]
            rows.append({
                "video_id": video_id,
                "video_path": str(video_path),
                "file_name": record["file_name"],
                "label": label,
                "event_timestamp": optional_float(record.get("time_of_event")),
                "alert_timestamp": optional_float(record.get("time_of_alert")),
                "dataset_source": "Nexar labeled training metadata",
                "event_relevance": "Unclear" if label else "Not applicable",
                "metadata_status": "metadata_present_video_present" if video_path.is_file() else "metadata_present_video_not_downloaded",
                "scene": record.get("scene", ""),
                "weather": record.get("weather", ""),
                "light_conditions": record.get("light_conditions", ""),
                "time_to_accident_raw": record.get("time_to_accident", ""),
            })

    # A provisional class-stratified video-level split. No windows are formed
    # here; source-session grouping is unavailable and must be revisited if found.
    rng = random.Random(args.seed)
    for label in (0, 1):
        group = [r for r in rows if r["label"] == label]
        rng.shuffle(group)
        n_train = int(len(group) * 0.70)
        n_val = int(len(group) * 0.15)
        for index, row in enumerate(group):
            row["split"] = "train" if index < n_train else "validation" if index < n_train + n_val else "test"

    # IDs are unique within the labeled training set; fail closed otherwise.
    ids = [str(row["video_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate video IDs found in labeled training metadata")

    fields = [
        "video_id", "video_path", "label", "split", "event_timestamp", "alert_timestamp",
        "dataset_source", "event_relevance", "metadata_status", "file_name", "scene",
        "weather", "light_conditions", "time_to_accident_raw",
    ]
    write_rows(output / "dataset_manifest.csv", rows, fields)
    write_rows(output / "split_manifest.csv", rows, fields)
    review_fields = ["video_id", "file_name", "label", "event_timestamp", "event_relevance", "review_status", "review_notes"]
    review = [{
        "video_id": row["video_id"],
        "file_name": row["file_name"],
        "label": row["label"],
        "event_timestamp": row["event_timestamp"],
        "event_relevance": row["event_relevance"],
        "review_status": "pending_human_review" if row["label"] else "not_applicable",
        "review_notes": "Positive event relevance is unknown; do not treat the default category as an annotation." if row["label"] else "",
    } for row in rows]
    write_rows(output / "event_relevance_review.csv", review, review_fields)

    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        counts[split] = {str(label): sum(r["split"] == split and r["label"] == label for r in rows) for label in (0, 1)}
    print(f"rows={len(rows)} unique_video_ids={len(set(ids))} split_counts={counts}")
    print(f"local_videos={sum(r['metadata_status'] == 'metadata_present_video_present' for r in rows)}")
    print(f"manifests={output}")


if __name__ == "__main__":
    main()
