"""Create review videos from cached detections/tracks without rerunning YOLO."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("stage4_two_architecture_comparison/data/manifests/split_manifest.csv"))
    parser.add_argument("--video-ids", nargs="+", required=True)
    parser.add_argument("--sample-rate", type=float, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else Path.cwd() / args.manifest
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        manifest = {row["video_id"]: row for row in csv.DictReader(handle)}
    out_dir = Path.cwd() / "stage4_two_architecture_comparison" / "outputs" / "videos" / "perception_validation" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for video_id in args.video_ids:
        if video_id not in manifest:
            raise KeyError(f"Video {video_id} is absent from the manifest")
        video_path = Path(manifest[video_id]["video_path"])
        tracks_path = run_dir / "tracks" / f"{video_id}.parquet"
        data = pd.read_parquet(tracks_path)
        by_frame = {int(k): group.to_dict("records") for k, group in data.groupby("frame_index")}
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"Cannot open video {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = out_dir / f"{video_id}_{args.sample_rate:g}fps_annotated.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.sample_rate, (width, height))
        if not writer.isOpened():
            cap.release()
            raise OSError(f"Cannot create annotated video {out_path}")

        frame_indices = sorted(by_frame)
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            timestamp = frame_index / max(cap.get(cv2.CAP_PROP_FPS), 1e-6)
            for row in by_frame[frame_index]:
                x1, y1, x2, y2 = (int(round(float(row[k]))) for k in ("x1", "y1", "x2", "y2"))
                track_id = str(row["track_id"])
                color = (0, 220, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{row['class_name']} ID {track_id} {float(row['confidence']):.2f}"
                cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{video_id}  t={timestamp:.2f}s", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
        cap.release()
        writer.release()
        print(out_path)


if __name__ == "__main__":
    main()
