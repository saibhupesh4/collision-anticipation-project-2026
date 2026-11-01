# Project Status — Stage 4

**Handoff:** Run the full-frame preprocessing pipeline on the balanced 400-video subset (200 positive, 200 negative). Do not start model training.

## User data and source audit

- `data/raw/positive/` contains 750 MP4s and `data/raw/negative/` contains 750 MP4s.
- The OpenCV full-decode integrity report lists 1,500/1,500 readable videos with no frame-count mismatch.
- The saved metadata manifests provide a fixed-seed, class-stratified 70/15/15 split: 525/112/113 videos per class (1,050/224/226 overall).
- Original videos and source metadata must remain unchanged. No downloads are needed.

## Preprocessing implementation

`src/common/full_dataset_preprocessing.py` runs YOLOv8n on every decoded frame, applies ByteTrack sequentially within each video, and writes per-frame detections, tracks, trajectory features, and nearest vehicle/person interaction features. CUDA:0 is explicit in the handoff command. The benchmark-selected inference settings are batch 16, FP16 (`quantize=16`), 640-pixel input, and an 800 MiB minimum VRAM reserve with an additional allocator margin. Video decoding, ByteTrack bookkeeping, feature-table construction, and Parquet I/O still use CPU; the video dataset is not loaded into VRAM.

The batch sweep tested 4–256 frames per batch on the same two clips (2,414 frames). Batch 16 was fastest safe at 94.1 end-to-end frames/sec; batch 256 exhausted GPU memory with only 300 MiB free. The chosen 400-video subset contains 450,882 frames, about an 80-minute estimate at benchmark speed. The estimate is approximate and the 2-clip sweep is not a full-dataset performance guarantee.

Subset membership is selected deterministically with seed 42 within each original split. Expected allocation is 140 positive + 140 negative train, 30 + 30 validation, and 30 + 30 test. No temporal windows cross video splits.

## Run command

From the project root in PowerShell:

```powershell
$py = 'C:\Users\saibh\anaconda3\envs\collision_env\python.exe'
& $py stage4_two_architecture_comparison\src\common\full_dataset_preprocessing.py `
  --device 0 --batch-size 16 --gpu-headroom-mb 800 `
  --subset-positive 200 --subset-negative 200 `
  2>&1 | Tee-Object -FilePath data\metadata\subset_200pos_200neg_preprocessing.log
```

The pipeline writes subset manifests to `data/splits/subset_200pos_200neg/`, and split-partitioned outputs to `data/detections/`, `data/tracks/`, and `data/features/`. Its status, failure, validation, config, and report files use the `subset_200pos_200neg_preprocessing_*` prefix in `data/metadata/`. It can resume completed videos after interruption.

## Validation and interpretation

- The pipeline checks per-frame index coverage and detection counts, frame totals, duplicate track IDs within a frame, bounding boxes, and non-finite feature values.
- Derivatives at track starts remain null because they are undefined; these values are not imputed. Interaction fields are null when no vehicle is present or the relative motion is unavailable.
- `time_to_conflict_image_sec`, proximity, closing rate, velocity, acceleration, and expansion are image-space proxies, not calibrated physical measurements or TTC.
- The test split is preprocessed but is not used for model fitting or feature selection. No model training is part of this handoff.

## Existing completed work

Phases 0–4 from the earlier 80-video smoke work, reports, and model checkpoints are preserved. The attempted full 1,500-video run was stopped before completion when the user requested a faster subset. Its partial preprocessing caches were cleaned; the resumable script and benchmark findings remain.
# Detailed function-by-function flow

See [reports/preprocessing_flow.md](reports/preprocessing_flow.md) for the exact CLI-to-function call order, helper responsibilities, per-video YOLO/ByteTrack lifecycle, feature definitions, resume behavior, output schemas, validation checks and benchmark flow.

See [reports/preprocessing_run_guide.md](reports/preprocessing_run_guide.md) for setup requirements, adjustable progress/ETA flags, reported performance metrics, and the annotation requirements for collision, detection and tracking evaluation metrics.

