import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.architecture_1.features import extract_features, make_horizon_examples


def row(video_id, timestamp, track_id, class_name, x1, x2, y1=100, y2=200):
    return {
        "video_id": video_id,
        "frame_index": int(timestamp * 30),
        "timestamp_sec": float(timestamp),
        "track_id": track_id,
        "class_id": 2 if class_name == "car" else 0,
        "class_name": class_name,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "confidence": 0.9,
        "track_age_sec": timestamp,
        "time_since_last_detection_sec": 0.2,
        "track_status": "active",
    }


class Architecture1FeatureTests(unittest.TestCase):
    def test_velocity_uses_actual_timestamps(self):
        tracks = pd.DataFrame([
            row("v1", 1.0, "2:1", "car", 50, 150),
            row("v1", 3.0, "2:1", "car", 178, 278),
        ])
        _, trajectory = extract_features(tracks, endpoint_sec=3.0, observation_sec=3.0)
        self.assertAlmostEqual(trajectory["speed_mean_car"], 0.05, places=6)

    def test_future_rows_do_not_change_features(self):
        before = pd.DataFrame([
            row("v1", 1.0, "2:1", "car", 50, 150),
            row("v1", 2.0, "2:1", "car", 114, 214),
        ])
        with_future = pd.concat([
            before,
            pd.DataFrame([row("v1", 2.2, "0:9", "person", 800, 1200, 0, 700)]),
        ], ignore_index=True)
        self.assertEqual(extract_features(before, 2.0), extract_features(with_future, 2.0))

    def test_horizon_examples_end_before_event(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            track_dir = Path(tmp)
            pd.DataFrame([
                row("positive", 1.0, "2:1", "car", 50, 150),
                row("positive", 3.0, "2:1", "car", 178, 278),
                row("positive", 3.6, "0:9", "person", 800, 1200, 0, 700),
            ]).to_parquet(track_dir / "positive.parquet", index=False)
            pd.DataFrame([row("negative", 1.0, "2:1", "car", 50, 150)]).to_parquet(track_dir / "negative.parquet", index=False)
            manifest = pd.DataFrame([
                {"video_id": "positive", "label": 1, "split": "train", "event_timestamp": 4.0},
                {"video_id": "negative", "label": 0, "split": "train", "event_timestamp": float("nan")},
            ])
            examples = make_horizon_examples(manifest, track_dir, horizons=(0.5,), observation_sec=3.0)
            positive = examples[examples.label == 1].iloc[0]
            self.assertLess(positive.endpoint_sec, positive.event_timestamp_sec)
            self.assertAlmostEqual(positive.endpoint_sec, 3.5)
            self.assertEqual(len(examples), 2)


if __name__ == "__main__":
    unittest.main()
