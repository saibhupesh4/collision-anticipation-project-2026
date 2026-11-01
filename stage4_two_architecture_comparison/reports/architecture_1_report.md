# Architecture 1 — Trajectory-Based Smoke Experiment

## Pipeline

```text
Dashcam video → shared YOLOv8n detections → shared ByteTrack tracks
             → 3-second image-space feature window
             → static XGBoost baseline / trajectory XGBoost / GRU
             → event probability at fixed future horizons
```

The fixed-horizon examples use one 3-second observation window per video and horizon. Positive examples end at `time_of_event - horizon`; negative examples use an endpoint aligned to the median event time of training positives. Only frames at or before the endpoint are used. The model never receives event time, alert time, weather, scene, lighting, filename, or video ID as a feature.

## Features and models

The static baseline uses active object counts, apparent box areas, and relative vehicle/pedestrian image-space proximity. The trajectory XGBoost model adds image-space velocity, acceleration proxies, box-area/height expansion, velocity variation, and relative closing patterns. No absolute image-center feature or `closest_dist_to_center_mean` feature is used. Relative pixel-derived rates are not physical speeds or TTC.

The GRU uses six steps over the preceding three seconds, with 17 aggregate motion/interaction features per step. Train-only normalization is used. The model has 1,697 learned parameters. XGBoost models use 80 depth-2 trees per horizon.

## Training and evaluation protocol

All 80 smoke-test videos were assigned by the provisional video-level split. At each horizon, the local sample provided 56 train videos (27 positive, 29 negative), 12 validation videos (6/6), and 12 test videos (7 positive, 5 negative). The test split was not used for threshold selection. Thresholds were chosen on validation data. These counts are far too small for stable performance estimates; confidence intervals and generalization claims are not justified.

## Measured smoke metrics

Average precision is reported on only 12 test videos per horizon. These values validate code execution and target plumbing only.

| Horizon | Static XGBoost AP | Trajectory XGBoost AP | GRU AP | Test videos |
|---:|---:|---:|---:|---:|
| 0.5 s | 1.000 | 0.863 | 0.904 | 12 |
| 1.0 s | 1.000 | 0.948 | 0.573 | 12 |
| 1.5 s | 0.968 | 0.909 | 0.751 | 12 |
| 2.0 s | 0.755 | 0.593 | 0.733 | 12 |
| 4.0 s | 0.740 | 0.874 | 0.728 | 12 |

Training time totaled 0.322 seconds across the ten XGBoost models and 0.767 seconds across the five GRU models on this tiny smoke run. These are not full training-cost estimates. Precision/recall at validation-selected thresholds, per-video scores, training history, and checkpoints are in the associated run directories.

## Shortcut and failure checks

The static model's largest feature importances were apparent car/truck box area and the number of tracks in the observation window. That pattern may reflect object scale, scene composition, or dataset-specific camera framing rather than useful anticipation. It warrants explicit shortcut analysis on a larger, better-balanced split. The previously flagged image-center-distance feature was excluded.

The sample has a known class-dependent scene distribution, and the detector has qualitative false detections and missed small/distant objects. No object-level labels exist to quantify perception error. The high AP values on 12 test clips should therefore not be interpreted as collision prediction performance.

## Conclusion

This smoke test confirms that leakage-checked event-relative feature generation, XGBoost training, GRU training, checkpoint writing, and per-horizon scoring run end to end. Static features scored higher than trajectory XGBoost at four horizons in this tiny sample; trajectory XGBoost scored higher at 4 seconds. The GRU did not improve consistently. No research conclusion about the value of trajectories or recurrent modeling is supported. False alarms per minute, warning lead time, end-to-end model latency, and confidence intervals are not measured here.

Run outputs:

- Features and GRU sequences: `experiments/runs/run_20261101_021212_architecture1_features/`
- Static and trajectory XGBoost: `experiments/runs/run_20261101_021240_architecture1_xgboost_smoke/`
- GRU: `experiments/runs/run_20261101_021248_architecture1_gru_smoke/`
