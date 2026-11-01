# Architecture 2 — Smoke Integration Report

**Date:** 2026-11-01  
**Scope:** 80-video pipeline smoke integration only. These measurements do not estimate generalization.

## Implementation

Architecture 2 reuses the same 5 FPS YOLOv8n/ByteTrack detections and tracks as Architecture 1. A constant-velocity Kalman predictor uses observed timestamps, uncertainty increases with prediction horizon and track gaps, and masked sparse optical flow with RANSAC estimates background camera translation, scale, and rotation. The risk stage compares predicted image-space tracks with a heuristic ego corridor and forms rule-only and Logistic Regression fusion scores. A no-camera-motion Logistic Regression model is included as an ablation.

Geometry is explicitly image-space only. `metric_geometry_valid` is false; physical distance, metric TTC, ego speed, braking, and verified ego-path threat cannot be inferred from these outputs. Camera-flow estimates may be poor in low-texture, moving-object-heavy, or strongly blurred frames.

## Pipeline status

- Full camera-motion extraction: 80/80 videos completed, using the shared detector boxes as flow masks.
- Horizon feature set: 400 examples (80 videos × 5 horizons), with the same provisional video-level smoke split as Architecture 1.
- Model runs: rule-only baseline plus five horizon models each for compensated fusion and no-camera ablation.
- Synthetic risk checks: 7 passed, covering crossing/receding paths, nonuniform timestamps, uncertainty across gaps, artificial global translation compensation, post-event exclusion, invalid metric geometry, and video ID scoping.

## Test-set average precision (smoke only)

Each horizon has 12 test videos (7 positive, 5 negative). Values are diagnostics on this tiny smoke partition.

| Horizon (s) | Rule only | Camera compensated fusion | No-camera ablation |
|---:|---:|---:|---:|
| 0.5 | 0.793 | 0.741 | 0.928 |
| 1.0 | 0.793 | 0.860 | 0.827 |
| 1.5 | 0.793 | 0.895 | 0.797 |
| 2.0 | 0.821 | 0.909 | 0.761 |
| 4.0 | 0.793 | 0.860 | 0.657 |

The small test set and video-level event-label limitations make these values unsuitable for choosing an architecture. The mixed ablation pattern is not evidence that camera compensation helps or hurts generally.

## Limitations and next steps

Nexar provides video-level collision-related labels, not object-level collision involvement, trajectories, ego motion, or metric road geometry. Some positive events may concern other road users rather than the dashcam vehicle. The event relevance review remains pending human review for positive clips. This prevents validated ego-centric false-positive and missed-threat analysis. Failure-case examples are pipeline diagnostics only; no object-level precision/recall is available.

Architecture 2 should be evaluated again on the expanded pilot, with an event-relevance review and larger untouched test set, before any comparison or recommendation. No additional videos were downloaded for this smoke phase.
