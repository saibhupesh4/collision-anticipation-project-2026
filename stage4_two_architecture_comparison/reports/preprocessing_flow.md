# Preprocessing pipeline: function-by-function flow

This document follows the implementation in [`full_dataset_preprocessing.py`](../src/common/full_dataset_preprocessing.py). It describes what is called, in what order, what each call reads/writes, and how the saved results are checked. It is for the current 400-video run (200 positive, 200 negative); the script also supports all 1,500 videos.

## Entry points

```text
PowerShell command
  └─ Python starts full_dataset_preprocessing.py
       └─ __main__ calls main()
            ├─ validate and write split manifests
            ├─ attach OpenCV integrity metadata
            ├─ optionally select the stratified subset
            ├─ initialize CUDA, YOLOv8n and ByteTrack
            ├─ process each video sequentially
            ├─ validate saved outputs
            └─ write final reports
```

The run command from the project root is:

```powershell
$py = 'C:\Users\saibh\anaconda3\envs\collision_env\python.exe'
& $py stage4_two_architecture_comparison\src\common\full_dataset_preprocessing.py `
  --device 0 --batch-size 16 --gpu-headroom-mb 800 `
  --subset-positive 200 --subset-negative 200 `
  2>&1 | Tee-Object -FilePath data\metadata\subset_200pos_200neg_preprocessing.log
```

This command uses the already downloaded videos and the existing local YOLO checkpoint. It does not train a model or modify source videos.

## Detailed call order

| Order | Function/call | Inputs and behavior | Outputs / next call |
|---:|---|---|---|
| 1 | `__main__ → main()` | Parses root, manifest, integrity report, local model path, image size (640), confidence (0.25), batch (16), device (CUDA 0), VRAM reserve, seed (42), subset counts and resume options. Resolves paths relative to project root. | Raises immediately if the checkpoint is absent or arguments are incompatible. |
| 2 | `make_split_outputs(root, manifest_path, seed)` | Reads the existing 1,500-row video-level manifest. Checks required columns, unique IDs, expected class-balanced seed-42 allocation (train 525/class, validation 112/class, test 113/class), legal split names and that every path resolves to a local file. It reuses the established split instead of reassigning clips. | Writes `data/splits/split_manifest.csv`, per-split CSV manifests, per-split ID text files and `split_config.json`; returns the validated dataframe. |
| 3 | integrity CSV load and merge inside `main()` | Reads `data/metadata/video_integrity_report.csv`; requires exactly 1,500 readable rows and unique filename stems. Joins FPS, decoded frame count, dimensions and duration to the manifest by video ID, one-to-one. | Enriched manifest used to estimate expected frames and timestamps. Missing metadata aborts before inference. |
| 4 | `make_stratified_subset(...)` (only with subset flags) | Deterministically shuffles each class *within its already assigned split* using seed 42. For each 200-class subset it selects 140 train, 30 validation and 30 test videos. Thus the subset is balanced and no video crosses splits. | Writes manifests and ID lists under `data/splits/subset_200pos_200neg/`, plus `subset_config.json`; returns the 400 selected rows. |
| 5 | device and memory setup inside `main()` | Chooses CUDA when requested/available. Checks the minimum 800 MiB reserve. Sets PyTorch's per-process memory fraction to leave the requested 800 MiB and an additional 800 MiB driver/display allowance. Enables FP16 for CUDA unless `--full-precision`; limits OpenCV threads and disables cuDNN autotuning (variable last-batch shapes make it costly). | Establishes the inference configuration before model load. CPU still performs video decode, ByteTrack bookkeeping, feature calculations and output I/O. Frames are sent to the GPU in batches; the whole dataset/video is not loaded into VRAM. |
| 6 | `YOLO(model_path)` and tracker YAML setup inside `main()` | Loads the local YOLOv8n checkpoint. Reads Ultralytics' ByteTrack configuration and sets `track_buffer=180` frames (about six seconds at 30 FPS). Detection class IDs are COCO person 0, bicycle 1, car 2, motorcycle 3, bus 5 and truck 7. | Writes `data/metadata/bytetrack_full_frame.yaml`; writes run parameters to `{scope}_preprocessing_config.json`. |
| 7 | `load_completed(status_path)` | If resume is enabled and the scope-specific summary CSV exists, loads the per-video status rows. A malformed/unreadable status file results in an empty cache, so clips will be retried. | A map of prior video records for the processing loop. |
| 8 | `verify_existing(row, expected_frames, batch_size, half_precision)` | For each clip, checks status is completed, processed frame count matches integrity metadata, batch and precision match this invocation, and all five output files still exist. | `True` means print `resume-skip`; otherwise that clip is processed again. The current handoff cleanup removed prior partial generated outputs, so the first handoff run starts fresh. |
| 9 | `process_video(...)` per selected video | Called once per video, sequentially. Receives the YOLO model, manifest row, expected metadata, output paths and inference/tracker settings. Calls `model.track(..., stream=True, persist=False, tracker=..., batch=16, device=0, quantize=16)` on the video. The Ultralytics predictor decodes batches, preprocesses/normalizes images and runs inference; the stream yields results in frame order and its tracking callback applies ByteTrack. After each frame it calls the `update_progress(frame_count)` callback supplied by `main()`. | For each decoded frame, creates one dense frame-index row even if no detections exist. For each selected-class detection it records box, class, confidence, ByteTrack ID, timestamp, track age and gap since last observation. IDs are scoped to each video and prefixed by class (`class_id:tracker_id`) to avoid cross-class ambiguity. |
| 10 | `derive_features(video_id, track_frame)` called at end of `process_video` | Sorts each track by timestamp/frame. Derives normalized box center, width, height and area; finite differences give image-space x/y velocity, acceleration magnitude, direction and wrapped direction change; log area change gives bounding-box expansion rate. Lateral movement is currently the x-direction image-space velocity proxy. Then, frame by frame, finds each person's nearest tracked car/bus/truck/motorcycle and computes normalized relative position/velocity, diagonal-normalized proximity, projected closing rate, threshold proximity indicators and `time_to_conflict_image_sec` when closing rate is positive. | Returns trajectory and interaction dataframes. These are image-coordinate proxies, not calibrated physical measurements, real-world speed/distance, collision probability or physical TTC. First observations and insufficient history remain null; no values are imputed. No vehicle in a frame yields `interaction_valid=false` and null nearest-vehicle measurements. |
| 11 | `atomic_parquet(...)` × five files in `process_video` | Writes to a temporary `.writing.parquet` file, then atomically renames it after a successful write. | Per-video Parquet outputs: detections, dense frame index, tracks, trajectory features and interactions. Atomic replacement avoids presenting half-written files as complete. |
| 12 | per-video record and `write_csv_atomic(...)` in `main()` | `process_video` catches inference errors and flags frame-count mismatches; outer `main()` also catches unexpected per-video exceptions. After each attempted clip, the result is merged into the cached records and summary is atomically rewritten. The frame callback updates `tqdm` with the configured `--progress-update-frames` granularity and `--progress-refresh-seconds` redraw interval. | Progress bar shows total frame progress, throughput and a frame-based ETA; per-video lines and progress postfix show status, detections, tracks and failures. `--no-progress` disables bars while keeping log lines. Failed/incomplete rows remain in summary and later appear in failures CSV; they are not silently skipped. |
| 13 | `validate_outputs(...)` after loop | Compares manifest, summary and integrity rows. Checks unique IDs and split disjointness, completed/failed/not processed counts, expected vs processed frames, per-frame index continuity, detection counts, valid box coordinates, duplicate `(frame_index, track_id)` pairs, feature row counts and unexpected non-finite values. Counts expected undefined derivatives and interactions without a vehicle separately. | Returns machine-readable checks and `pass` boolean. A failed video or failed invariant prevents a clean pass. |
| 14 | final `atomic_json`, `write_csv_atomic`, Markdown write | Saves validation JSON, writes all non-completed summary records to failure CSV, writes the human-readable report, and prints a compact JSON summary to terminal. | The final report states scope, counts, paths and image-space interpretation limitations. |

## Function reference

### `full_dataset_preprocessing.py`

- `atomic_parquet(frame, path)`: creates parent directories, writes a temporary Parquet, atomically replaces target.
- `atomic_json(payload, path)`: JSON serializes with NaN prohibited, writes temporary file and atomically replaces target.
- `write_csv_atomic(frame, path)`: atomic CSV writer used for manifests, status and failures.
- `make_split_outputs(...)`: checks and exports the existing full video split.
- `make_stratified_subset(...)`: selects exact per-class counts while retaining the original split groups.
- `wrap_angle(value)`: normalizes an angle delta to `[-π, π)` for direction-change features.
- `derive_features(video_id, tracked)`: computes per-track motion and per-person/nearest-vehicle interaction features.
- `load_completed(status_path)`: loads resume state from the prior summary file.
- `verify_existing(...)`: determines whether a completed cache entry is compatible and complete.
- `process_video(...)`: runs batched YOLO + frame-ordered ByteTrack on one source clip, builds rows, derives features and persists five outputs.
- `validate_outputs(...)`: aggregates coverage, uniqueness, frame, detection, box, track and feature checks.
- `main()`: orchestrates argument parsing, data checks, subset selection, GPU/model setup, sequential clip loop, resume, validation and reports.

### `benchmark_gpu_batch_sizes.py` (already used to choose batch 16)

- `sample_gpu()`: queries `nvidia-smi` once per second for GPU utilization and memory; returns `None` if the query fails.
- `main()`: loops batch candidates, launches the preprocessing CLI as a subprocess on fixed positive/negative sample videos, records samples and logs, reads per-video summaries, calculates throughput and reserve compliance, stops at the first failed/unsafe size and selects the fastest successful safe candidate.

The measured sample was two videos (2,414 frames), so its 94.1 end-to-end frames/second and estimated runtime are projections. The 400-video selection has about 450,882 expected frames, projecting to roughly 80 minutes at that measured throughput; actual duration depends on clip decode, detections, GPU state and disk speed.

## Outputs and row contents

All per-video files are partitioned by the original split and named with the manifest video ID:

| Location | Meaning |
|---|---|
| `data/splits/` | Full split manifest, train/validation/test manifests and ID lists. Subset equivalents are in `data/splits/subset_200pos_200neg/`. |
| `data/detections/{split}/{video_id}.parquet` | One row per selected-class detection: frame/time, box (`x1,y1,x2,y2`), class, confidence, tracker ID and track-age/gap fields. |
| `data/detections/frames/{split}/{video_id}.parquet` | One row per decoded frame including frames with zero detections: frame/time, detection count and inference milliseconds. |
| `data/tracks/{split}/{video_id}.parquet` | Tracked detection rows with track ID and box/class/confidence fields. |
| `data/features/trajectories/{split}/{video_id}.parquet` | Per-track observation, normalized geometry and image-space motion/expansion features. |
| `data/features/interactions/{split}/{video_id}.parquet` | One row per tracked person per frame, with nearest vehicle interaction estimates where one is tracked. |
| `data/metadata/subset_200pos_200neg_preprocessing_*` | Scope config JSON, incremental summary CSV, failures CSV, validation JSON and final report Markdown; console log is saved separately by the handoff command. |

The full-data invocation uses `full_preprocessing_*` metadata names. The status summary includes video/split/category, expected and processed counts, dimensions, duration, row counts, runtime, inference time, batch/precision, peak CUDA memory and each output path.

## What validation does and does not establish

The validation report confirms pipeline coverage and structural consistency; it does not measure detector accuracy or prove that every true object was detected. A clean `pass` requires all selected manifest videos to complete and the listed structural checks to pass. The subset run validates its 400-video scope and records the 1,500-video overall source manifest count separately. Run without subset flags to cover all 1,500 videos. No model training or test-set-based feature selection is performed.

## Companion handoff documents

- [`README.md`](../README.md): run command, benchmark and setup notes.
- [`preprocessing_run_guide.md`](preprocessing_run_guide.md): environment prerequisites, adjustable flags, ETA/performance metrics and evaluation metric boundaries.
- [`PROJECT_STATUS.md`](../PROJECT_STATUS.md): current handoff state and user decisions.
- [`methodology.md`](methodology.md): project methodology notes.
