import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.architecture_2.camera_motion import estimate_camera_motion
from src.architecture_2.conflict_analysis import _camera_compensated, analyze_conflicts
from src.architecture_2.ego_corridor import corridor_bounds, image_space_geometry_status
from src.architecture_2.trajectory_prediction import predict_track


def track_rows(xs, times=(0.0, 0.5, 1.0), track_id="0:1", class_name="person", y=0.78):
    rows = []
    for timestamp, x in zip(times, xs):
        center = x * 1280.0
        rows.append({
            "video_id": "v1", "frame_index": int(timestamp * 5), "timestamp_sec": timestamp,
            "track_id": track_id, "bytetrack_id": 1, "class_id": 0 if class_name == "person" else 2,
            "class_name": class_name, "x1": center - 15, "x2": center + 15,
            "y1": (y - 0.15) * 720, "y2": y * 720, "confidence": 0.9,
            "track_age_sec": timestamp, "time_since_last_detection_sec": 0.2,
            "track_status": "active",
        })
    return pd.DataFrame(rows)


class Architecture2RiskTests(unittest.TestCase):
    def test_crossing_into_corridor_scores_above_clearing_and_receding(self):
        crossing = analyze_conflicts(track_rows([0.2, 0.35, 0.5]), 1.0, 0.5)
        clearing = analyze_conflicts(track_rows([0.7, 0.45, 0.2]), 1.0, 0.5)
        receding = analyze_conflicts(track_rows([0.85, 0.9, 0.95]), 1.0, 0.5)
        self.assertGreater(crossing["rule_risk"], clearing["rule_risk"])
        self.assertGreater(crossing["rule_risk"], receding["rule_risk"])

    def test_nonuniform_timestamps_drive_prediction(self):
        times = (0.0, 0.4, 1.3)
        xs = [0.2 + 0.1 * t for t in times]
        prediction = predict_track(track_rows(xs, times=times), endpoint_sec=1.3, horizon_sec=1.0)
        self.assertTrue(prediction["valid"])
        self.assertAlmostEqual(prediction["vx"], 0.1, delta=0.04)

    def test_tracking_gap_increases_uncertainty(self):
        tracks = track_rows([0.2, 0.35], times=(0.0, 0.5))
        fresh = predict_track(tracks, endpoint_sec=0.5, horizon_sec=0.5)
        gapped = predict_track(tracks, endpoint_sec=1.5, horizon_sec=0.5)
        self.assertGreater(gapped["gap_sec"], fresh["gap_sec"])
        self.assertGreater(gapped["sigma"], fresh["sigma"])

    def test_artificial_camera_translation_is_compensated(self):
        rng = np.random.default_rng(7)
        previous = rng.integers(0, 256, (240, 320), dtype=np.uint8)
        transform = np.float32([[1, 0, 6], [0, 1, -4]])
        current = cv2.warpAffine(previous, transform, (320, 240))
        estimate = estimate_camera_motion(previous, current)
        self.assertTrue(estimate["valid"])
        self.assertAlmostEqual(estimate["dx_px"], 6.0, delta=1.0)
        self.assertAlmostEqual(estimate["dy_px"], -4.0, delta=1.0)
        tracks = track_rows([0.4, 0.4], times=(0.0, 0.2))
        tracks.loc[tracks.frame_index == 1, ["x1", "x2"]] += 6.0
        camera = pd.DataFrame([
            {"frame_index": 0, "timestamp_sec": 0.0, "valid": False, "dx_px": 0.0, "dy_px": 0.0, "inlier_ratio": 0.0, "uncertainty": 1.0},
            {"frame_index": 1, "timestamp_sec": 0.2, "valid": True, "dx_px": 6.0, "dy_px": 0.0, "inlier_ratio": 0.9, "uncertainty": 0.1},
        ])
        stabilized, _ = _camera_compensated(tracks, camera)
        centers = (stabilized.x1 + stabilized.x2) / 2
        self.assertLess(float(centers.max() - centers.min()), 1e-5)

    def test_post_event_rows_do_not_change_conflict_score(self):
        before = track_rows([0.2, 0.35, 0.5])
        future = track_rows([0.2, 0.35, 0.5, 0.99], times=(0.0, 0.5, 1.0, 1.5), track_id="0:9")
        combined = pd.concat([before, future.iloc[[-1]]], ignore_index=True)
        self.assertEqual(analyze_conflicts(before, 1.0, 0.5), analyze_conflicts(combined, 1.0, 0.5))

    def test_invalid_metric_geometry_never_emits_physical_ttc(self):
        status = image_space_geometry_status()
        self.assertFalse(status["metric_geometry_valid"])
        self.assertIsNone(status["distance_m"])
        self.assertIsNone(status["ttc_s"])
        self.assertIsNone(corridor_bounds(0.2))

    def test_video_ids_are_assigned_to_only_one_split(self):
        path = Path("data/manifests/split_manifest.csv")
        manifest = pd.read_csv(path, dtype={"video_id": str})
        self.assertEqual(manifest.video_id.nunique(), len(manifest))
        self.assertEqual(manifest.groupby("video_id").split.nunique().max(), 1)


if __name__ == "__main__":
    unittest.main()
