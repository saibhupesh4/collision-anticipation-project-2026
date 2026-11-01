# Temporal Window Generation Specification

## Inputs and split boundary

Read the video-level `split_manifest.csv` first. Every window inherits its source video's single split. Do not create windows and then split them. Never move a video or a window across splits after model tuning begins.

Use actual elapsed video timestamps for observation endpoints. The provided event annotations are elapsed seconds from the beginning of the clip. The annotation's frame-boundary convention is unspecified; do not imply frame-level timing precision.

## Target for horizon `h`

At an observation endpoint `t`, label the window positive when a positive event is annotated and `0 < time_of_event - t <= h`; otherwise label it negative. The endpoint and every input frame must precede the event. Exclude any observation window containing a frame at or after `time_of_event`. For negative videos, all eligible observation times have target 0.

Evaluate each requested horizon (0.5, 1.0, 1.5, 2.0, and 4.0 seconds) as a separate target/task. The 0.5/1.0/1.5-second values match the official challenge horizons; 2.0/4.0 seconds are project extensions and must be reported separately. Primary metric comparisons use predictions at the same event-relative horizon on the same held-out videos.

The `time_of_alert` timestamp is stored only as a human-annotation reference for secondary analysis. It is not a feature and does not change the event-horizon target. Report its subjectivity and never describe it as an actual driver warning.

## Correlation and evaluation

Temporal windows from one video are correlated. Confidence intervals and bootstrap resampling must operate at video level, not window level. When reporting fixed-horizon AP, aggregate one score per video/horizon according to a documented prediction rule. Report warning lead and false alarms per minute separately from frame/window classification scores.

All feature scaling, imputation, feature selection, calibration, and warning thresholds are fit using training and validation videos only. The test split is opened once for final evaluation. Metadata columns are audit fields only and are excluded from model inputs.

## Smoke-test rule

The current 80 videos validate reading, extraction, feature generation, and label code. Any temporary smoke split or smoke metrics must be labeled as integration diagnostics, never as a generalization estimate. No overlapping windows from one video may appear across split boundaries.
