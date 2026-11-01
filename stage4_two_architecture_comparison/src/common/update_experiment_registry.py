"""Aggregate per-run records and measured smoke metrics into one registry."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


FIELDS = [
    "run_id", "architecture", "model", "dataset_subset", "split", "configuration", "seed",
    "completion_status", "videos", "processing_seconds", "sampled_frames", "detection_rows",
    "tracked_rows", "checkpoint_location", "report_location", "notes",
]


def main() -> None:
    runs_dir = Path("stage4_two_architecture_comparison/experiments/runs").resolve()
    output = runs_dir / "experiment_registry.csv"
    records: dict[str, dict[str, object]] = {}
    if output.exists():
        with output.open("r", newline="", encoding="utf-8-sig") as handle:
            records = {r["run_id"]: r for r in csv.DictReader(handle) if r.get("run_id")}

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")):
        run_id = run_dir.name
        registry_path = run_dir / "experiment_registry.csv"
        if registry_path.exists():
            with registry_path.open("r", newline="", encoding="utf-8-sig") as handle:
                local = next(csv.DictReader(handle), {})
        else:
            local = {}
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        summary_path = run_dir / "video_summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path, dtype={"video_id": str})
            videos = len(summary)
            processing = float(summary["processing_seconds"].sum()) if "processing_seconds" in summary else 0.0
            sampled = int(summary["sampled_frames"].sum()) if "sampled_frames" in summary else 0
            detections = int(summary["detection_rows"].sum()) if "detection_rows" in summary else 0
            tracks = int(summary["tracked_rows"].sum()) if "tracked_rows" in summary else 0
            status = "completed" if summary.get("status", pd.Series(dtype=str)).eq("ok").all() else "completed_with_failures"
        else:
            videos, processing, sampled, detections, tracks = 0, 0.0, 0, 0, 0
            status = str(local.get("completion_status", "failed"))
        notes = ""
        issue_path = run_dir / "known_issue.md"
        if issue_path.exists():
            notes = issue_path.read_text(encoding="utf-8").replace("\n", " ").strip()
        record = {
            "run_id": run_id,
            "architecture": local.get("architecture", config.get("architecture", "shared perception")),
            "model": local.get("model", config.get("model", "")),
            "dataset_subset": local.get("dataset_subset", "80-video smoke subset" if videos <= 80 else "manifest subset"),
            "split": local.get("split", "smoke_test_only"),
            "configuration": str(config_path) if config_path.exists() else local.get("configuration", ""),
            "seed": local.get("seed", config.get("seed", 42)),
            "completion_status": status,
            "videos": videos,
            "processing_seconds": round(processing, 4),
            "sampled_frames": sampled,
            "detection_rows": detections,
            "tracked_rows": tracks,
            "checkpoint_location": local.get("checkpoint_location", config.get("model", "")),
            "report_location": str(summary_path) if summary_path.exists() else str(run_dir / "error.log"),
            "notes": notes or records.get(run_id, {}).get("notes", ""),
        }
        records[run_id] = record

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records[k] for k in sorted(records))
    print(f"registered_runs={len(records)} path={output}")


if __name__ == "__main__":
    main()
