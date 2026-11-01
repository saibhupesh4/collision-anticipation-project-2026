# Early Collision Anticipation — Stage 4

This stage implements two collision anticipation approaches. The full-frame preprocessing code is ready; the current handoff is a deterministic, balanced 400-video subset (200 positive and 200 negative) so it finishes sooner. All 1,500 source videos passed the OpenCV integrity check. Read [PROJECT_STATUS.md](PROJECT_STATUS.md) before continuing.

For the complete preprocessing call graph, each helper's role, formulas, cache/resume rules, and output schemas, see [the function-by-function preprocessing flow](reports/preprocessing_flow.md).

For environment requirements, adjustable options, live ETA and performance metrics, and which metrics require separate model evaluation, see the [preprocessing run guide](reports/preprocessing_run_guide.md).

## Full-dataset preprocessing

Run from the project root in PowerShell with the existing `collision_env` Python. This processes 200 positive and 200 negative videos, keeps their original video-level train/validation/test assignments, and saves progress to a subset-specific summary:

```powershell
$py = 'C:\Users\saibh\anaconda3\envs\collision_env\python.exe'
& $py stage4_two_architecture_comparison\src\common\full_dataset_preprocessing.py `
  --device 0 --batch-size 16 --gpu-headroom-mb 800 `
  --subset-positive 200 --subset-negative 200 `
  --progress-update-frames 30 --progress-refresh-seconds 1 `
  2>&1 | Tee-Object -FilePath data\metadata\subset_200pos_200neg_preprocessing.log
```

The pipeline reuses the existing seed-42 video-level split, processes one video at a time, runs YOLOv8n on every decoded frame in CUDA FP16 batches, and applies ByteTrack updates in frame order. Ultralytics performs image conversion, resize/letterbox, and input normalization. CPU decoding, tracker bookkeeping, and Parquet I/O remain CPU-side; decoded frames are not all loaded into GPU memory. A PyTorch allocator cap reserves the requested 800 MiB plus an additional 800 MiB margin for display/driver use. cuDNN autotuning is disabled because variable final-batch shapes made its tuning overhead slow some clips. Completed videos are cached and skipped on restart when their settings and frame counts match.

The batch-size sweep used the same two clips (`00822`, `01924`; 2,414 frames) for every candidate. Batch 16 was fastest end to end at **94.1 frames/sec**, with **396 MiB** peak sampled GPU memory and 77% maximum sampled utilization. Batch 256 ran out of memory with only 300 MiB free and is excluded. The selected 400-video subset contains **450,882 frames**, projecting to **about 80 minutes** at the measured rate. This estimate is based on two clips; actual runtime will vary.

| Batch | End-to-end frames/sec | Peak sampled GPU memory | Result |
|---:|---:|---:|---|
| 4 | 58.0 | 174 MiB | Safe |
| 8 | 80.8 | 236 MiB | Safe |
| **16** | **94.1** | **396 MiB** | **Fastest safe** |
| 32 | 90.3 | 646 MiB | Safe |
| 64 | 83.1 | 1,094 MiB | Safe |
| 96 | 72.8 | 1,644 MiB | Safe |
| 128 | 80.6 | 2,022 MiB | Safe |
| 160 | 83.2 | 2,574 MiB | Safe |
| 192 | 83.3 | 2,968 MiB | Safe |
| 224 | 77.4 | 3,054 MiB | Safe, near reserve |
| 256 | — | OOM | Failed with 300 MiB free |

These measurements are from two clips and include process startup/output overhead. Results are recorded in this README; benchmark logs can be regenerated with `src/common/benchmark_gpu_batch_sizes.py`.

Outputs are partitioned by split: `data/splits/`, `data/detections/`, `data/tracks/`, and `data/features/`. The dense per-frame index is in `data/detections/frames/`. Progress, failures, and final validation are saved in `data/metadata/full_preprocessing_summary.csv`, `full_preprocessing_failures.csv`, and `full_preprocessing_validation.json`.

## Architecture 1

Uses image-space tracked-object trajectories and motion features with static XGBoost, temporal XGBoost, and a small GRU. See [architecture_1_report.md](reports/architecture_1_report.md).

## Architecture 2

Uses shared tracks, timestamp-aware Kalman predictions, optical-flow camera-motion compensation, an image-space ego corridor, risk fusion, and a no-camera ablation. Metric geometry is not available. See [architecture_2_report.md](reports/architecture_2_report.md).

## Reproducing smoke runs

From the project root in Windows PowerShell, use the existing environment Python directly (Conda activation currently fails in this session):

```powershell
$py = 'C:\Users\saibh\anaconda3\envs\collision_env\python.exe'
& $py stage4_two_architecture_comparison\src\architecture_1\prepare_features.py
& $py stage4_two_architecture_comparison\src\architecture_1\training.py
& $py stage4_two_architecture_comparison\src\architecture_1\train_gru.py
```

The commands above illustrate Architecture 1 entry points; supply the run paths/options shown by each script's `--help` for a fresh run. Architecture 2 feature generation and training use the preserved perception and camera-motion caches:

```powershell
& $py stage4_two_architecture_comparison\src\architecture_2\prepare_features.py `
  --perception-run stage4_two_architecture_comparison\experiments\runs\run_20261101_015729_shared_5fps `
  --camera-run stage4_two_architecture_comparison\experiments\runs\run_20261101_021646_architecture2_camera_motion
& $py stage4_two_architecture_comparison\src\architecture_2\training.py `
  --features stage4_two_architecture_comparison\experiments\runs\run_20261101_022425_architecture2_features\features.parquet
```

## Research safeguards

- All smoke results use only 80 videos and a 12-video test partition; they do not support generalization claims or an architecture recommendation.
- Splits are at video level; source-session grouping is unavailable.
- Metadata, IDs, `time_of_alert`, and post-event frames are excluded from model inputs.
- The dataset has video-level event labels, not object-level collision involvement or ground-truth trajectories.
- Image-space corridor/risk quantities do not represent physical distance or calibrated TTC.
- Do not download additional videos until the expanded-pilot selection, exact byte/time estimate, and storage reserve are reported and user approval is obtained.

## Reports

- [Dataset audit](reports/dataset_audit.md)
- [Window generation specification](reports/window_generation_spec.md)
- [Perception report](reports/phase2_perception_report.md)
- [Architecture 1](reports/architecture_1_report.md)
- [Architecture 2](reports/architecture_2_report.md)
- [Expanded pilot download proposal](reports/expanded_pilot_download_proposal.md)
