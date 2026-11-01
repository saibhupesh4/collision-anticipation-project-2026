# Running preprocessing: setup, controls and metrics

This guide is for `src/common/full_dataset_preprocessing.py`. It prepares detections, tracks and features; it does **not** fit a model. Do not interpret preprocessing speed, GPU use, detection counts or tracking counts as accuracy, precision, recall or F1.

## Before starting

Run commands from the project root (`C:\Users\saibh\OneDrive\Desktop\cv project`) in PowerShell. The current handoff command uses this existing environment Python:

```powershell
$py = 'C:\Users\saibh\anaconda3\envs\collision_env\python.exe'
```

Required local inputs:

- Videos under `data/raw/positive/` and `data/raw/negative/` (do not move or edit them).
- Integrity report: `data/metadata/video_integrity_report.csv`.
- Existing video split manifest: `stage4_two_architecture_comparison/data/manifests/split_manifest.csv`.
- Local checkpoint: `stage4_two_architecture_comparison/models/shared/yolov8n.pt`.
- Environment packages: Python, OpenCV (`cv2`), NumPy, pandas, PyTorch with CUDA support for GPU use, Ultralytics, PyYAML, tqdm, and a Parquet engine (`pyarrow` or `fastparquet`).
- NVIDIA driver / `nvidia-smi` for GPU diagnostics. The script can use CPU with `--device cpu`, but the recommended handoff uses CUDA device 0.

Optional preflight checks:

```powershell
& $py -c "import cv2, numpy, pandas, torch, ultralytics, yaml, tqdm, pyarrow; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
nvidia-smi
```

If import check fails due to missing `pyarrow`, install it into the existing environment with `& $py -m pip install pyarrow`. Do not install a CPU-only PyTorch build over a working CUDA build.

## Recommended: 200 positive + 200 negative

This preserves split membership, using 140 positive/140 negative train, 30/30 validation and 30/30 test. It shows a tqdm frame counter with live ETA and per-video completion lines. Progress refresh interval and update granularity are adjustable.

```powershell
$py = 'C:\Users\saibh\anaconda3\envs\collision_env\python.exe'
& $py stage4_two_architecture_comparison\src\common\full_dataset_preprocessing.py `
  --device 0 --batch-size 16 --gpu-headroom-mb 800 `
  --subset-positive 200 --subset-negative 200 `
  --progress-update-frames 30 --progress-refresh-seconds 1 `
  2>&1 | Tee-Object -FilePath data\metadata\subset_200pos_200neg_preprocessing.log
```

## Full 1,500-video run

Omit subset flags. Start this only if you intend to process all 1,500 videos.

```powershell
$py = 'C:\Users\saibh\anaconda3\envs\collision_env\python.exe'
& $py stage4_two_architecture_comparison\src\common\full_dataset_preprocessing.py `
  --device 0 --batch-size 16 --gpu-headroom-mb 800 `
  --progress-update-frames 30 --progress-refresh-seconds 1 `
  2>&1 | Tee-Object -FilePath data\metadata\full_preprocessing.log
```

## Settings you can adjust

Use `& $py stage4_two_architecture_comparison\src\common\full_dataset_preprocessing.py --help` to see the authoritative options.

| Option | Default / example | Effect |
|---|---|---|
| `--subset-positive N --subset-negative N` | `200` and `200` | Processes deterministic class-balanced subsets while retaining original split assignments. Omit both for all 1,500. |
| `--batch-size N` | `16` | Consecutive frames per inference batch. The local two-video sweep selected 16 as fastest safe; memory/throughput may differ on your machine. |
| `--device DEVICE` | `auto`; recommended `0` | `0` explicitly selects CUDA GPU 0; `auto` uses CUDA if available, otherwise CPU. |
| `--gpu-headroom-mb N` | `800` (minimum) | Requested free-VRAM reserve. PyTorch's allocator cap also leaves an extra 800 MiB margin for driver/display allocations. |
| `--imgsz N` | `640` | YOLO inference image size. Larger values can increase detail and VRAM/time. |
| `--confidence FLOAT` | `0.25` | Detection confidence threshold; changing it affects which detections, tracks and derived features are produced. |
| `--progress-update-frames N` | `30` | Calls progress update after each N newly processed frames; smaller values report more often. |
| `--progress-refresh-seconds FLOAT` | `1.0` | Minimum interval between visible tqdm refreshes. |
| `--no-progress` | off | Turns off tqdm bars; per-video completion log lines remain. |
| `--no-resume` | off | Reprocesses videos even if a compatible completed cache exists. Normally leave off so completed videos are reused. |
| `--full-precision` | off | Disables CUDA FP16 inference; normally leave off on the RTX 3050. |
| `--seed N` | `42` | Controls deterministic subset selection and recorded split configuration. |

`--progress-update-frames` controls the callback frequency; `--progress-refresh-seconds` controls how often tqdm redraws. `--batch-size`, image size, confidence, precision and device are also configurable. Changing batch size or precision causes cached videos from different settings to be reprocessed rather than silently reused.

## What appears during and after the run

- Global progress: processed frames / selected expected frames, elapsed time, frames/sec and tqdm's remaining-time estimate.
- Per-video line: video ID, status, frames decoded vs expected, detection rows, track rows and processing seconds.
- Final validation JSON: completion/failure counts, frame and structural checks plus performance metrics (frames/sec, completed videos/hour for this invocation, average video time, YOLO inference FPS/latency, detections/frame, checkpoint file size, peak/mean PyTorch GPU allocation and reserved memory).
- Summary CSV: per-video FPS, expected/processed frames, output row counts, processing/inference seconds, configured batch/precision and peak CUDA allocation.
- Failure CSV: every failed or incomplete video, with its error/status.

The ETA is a throughput projection and will vary with video length/content, GPU load, decode speed, file cache and disk writes. GPU allocation metrics are PyTorch allocator metrics; they are not the same as total memory shown by `nvidia-smi`.

## Collision anticipation evaluation metrics

These are the important **model evaluation** metrics for collision anticipation. They are not runtime/preprocessing metrics. The current source annotations provide positive/negative video labels, but the preprocessing pipeline does not have per-frame collision labels, collision timestamps, or model prediction scores. So it must not print made-up AP/F1/TTA values. The final preprocessing report provides ETA, throughput, processing times, data checks and GPU memory; the following metrics belong in a separate evaluation report after model training and prediction.

| Metric | Meaning | Required evaluation data / protocol |
|---|---|---|
| AP (Average Precision) | Precision-recall area for collision prediction at one defined anticipation horizon. | Ground-truth event labels at that horizon and continuous prediction scores. |
| mAP across horizons | Mean AP over the configured prediction horizons. | AP computed independently for each horizon; report the actual horizon list and averaging rule. |
| Precision | Of the predictions called collision, fraction that are true positives. | Binary decisions using a threshold selected using validation data, plus held-out labels. |
| Recall | Fraction of actual collision cases detected. | Held-out labels and a defined decision threshold. |
| F1-score | Harmonic mean of precision and recall. | Precision and recall at the same threshold and horizon. |
| PR-AUC | Area under the precision-recall curve. | Ground-truth labels and score ranking; particularly useful when collision positives are rare. |
| ROC-AUC | Area under the receiver operating characteristic curve. | Ground-truth labels and continuous scores; can look optimistic for rare positives, so report PR-AUC too. |
| Confusion matrix | Counts of TP, TN, FP and FN. | Binary threshold and a declared evaluation unit (frame, window or video). |
| mTTA (Mean Time-to-Accident) | Mean lead time between the first qualifying correct warning and the annotated accident. | Exact accident timestamps and a stated rule for what counts as a qualifying correct warning. Define handling of missed accidents. |
| FAR (False Alarm Rate) | Rate of false collision warnings on negative/non-event data. | Negative clips and an explicit unit, such as false alarms per video or per hour, plus alarm aggregation/debounce rules. |
| EDR (Early Detection Rate) | Fraction of collision events detected at least the required lead time before the event. | Accident timestamps, prediction traces and a stated minimum anticipation lead time. |
| P80R | Precision when operating at 80% recall. | Continuous scores and a threshold chosen on validation to reach 80% recall; report test precision at that frozen threshold. |

Before implementing/running this evaluation, provide or construct trustworthy event-time annotations and per-frame/window labels for positive clips, confirm negative clips are genuinely non-event examples over their full duration, specify anticipation horizons, define the evaluation unit and FAR units, and save model score outputs. Split by video as already done. Fit models and select features/thresholds using train/validation only; use test once for final metrics. mTTA's precise convention varies across papers, so the warning and missed-event rules must be written into the evaluation protocol. Keep the image-space TTC proxy from preprocessing separate: it is not an accident label or ground-truth mTTA.

The prediction-time evaluator should emit per horizon: AP, precision, recall, F1, PR-AUC, ROC-AUC and confusion matrix, alongside mAP; event-level mTTA, FAR and EDR; and P80R with its validation-selected threshold. Each metric should include its horizon, threshold, positive/event counts and evaluation unit in the report. No such evaluator is run by the preprocessing command.

## Detection, tracking and real-time evaluation

| Area | Metrics to report | Labels needed | Available from this preprocessing run? |
|---|---|---|---|
| YOLO object detection | Precision, recall, mAP@0.5, mAP@0.5:0.95; IoU is the box overlap measure used to match predictions to annotated boxes. | Frame-level ground-truth bounding boxes and classes on an evaluation set. Nexar video-level positive/negative labels do not contain boxes. | No accuracy metrics. The run does save detections and inference timing, but cannot calculate box precision/recall/mAP/IoU without labeled boxes. |
| ByteTrack tracking | HOTA, MOTA, IDF1, ID switches, ID precision/recall and track fragmentation. | Ground-truth object boxes/classes and persistent object identities across frames. | No tracking accuracy metrics. Tracker IDs are generated predictions and cannot serve as their own ground truth. |
| Early warning / collision anticipation | mTTA, EDR, FAR (including false alarms/minute if that is the chosen unit), detection delay and warning stability. | Per-event accident timestamps and frame/window/video prediction traces; protocol must define minimum lead time and alarm aggregation. | Not yet. The image-space time-to-conflict feature is only a proxy; it is neither ground-truth accident time nor mTTA. |
| Runtime / resource use | Detection inference FPS, end-to-end pipeline FPS, inference latency, processing time per video, peak GPU memory, model size; optionally parameters/FLOPs. | Timings and hardware; checkpoint file for model size; model graph/summary for parameters and FLOPs. | Partially. The report records YOLO inference FPS/mean inference milliseconds per frame, per-video pipeline FPS/time, checkpoint file size and peak PyTorch GPU allocation/reservation. It does not currently calculate parameter count/FLOPs or calibrated online warning latency. |
| Research robustness | Brier score, expected calibration error, confidence intervals, per-condition AP, ablations; trajectory prediction work may also report ADE/FDE/miss rate when that task and labels are defined. | Calibrated prediction scores, enough labeled evaluation examples, condition metadata, ablation runs, or future trajectory ground truth as applicable. | No. These require separate labeled experiments. |

Keep **detection inference FPS** (YOLO-only timing) separate from **end-to-end pipeline FPS** (decode + inference + ByteTrack + features + Parquet writes). The live tqdm rate is the run's processed-frame throughput; the final report's per-video pipeline FPS aggregates clip processing times, so startup and between-video report-writing overhead are excluded. Offline preprocessing throughput is not evidence of real-time warning latency. Likewise, model file size is not GPU memory, and generated track IDs are not tracking ground truth.

## Recommended final results table

Only populate a cell after measuring it with the corresponding labels and a written protocol. An initial comparison can use the rows below and columns such as AP@0.5s, AP@1.0s, AP@1.5s, mean AP, F1, mTTA, FAR/min and end-to-end FPS. Suggested model rows: detection only; detection + ByteTrack; trajectory + XGBoost; trajectory + GRU; Architecture 2 risk fusion. Add detection mAP and tracking HOTA/IDF1 in separate tables because their annotation requirements differ from collision classification.

Prioritize the user's initial ten collision metrics: AP at 0.5/1.0/1.5 seconds, mean AP across those horizons, precision, recall, F1, confusion matrix, mTTA and false alarms per minute. The horizon list, event definition, decision threshold, FAR denominator and minimum warning lead time must be fixed before comparing models. If the dataset lacks event timestamps or per-frame labels, report these as unavailable until they are annotated; do not infer them from the video-level class label.

## Outputs and resume files

Subset outputs use `data/splits/subset_200pos_200neg/` and split-partitioned Parquet under `data/detections/`, `data/tracks/`, and `data/features/`. Run status and final artifacts are in `data/metadata/subset_200pos_200neg_preprocessing_summary.csv`, `..._failures.csv`, `..._validation.json`, `..._config.json` and `..._report.md`. Full scope uses the `full_preprocessing_*` prefix. A completed video is skipped on restart only when expected frame count, batch size, precision and all five output files match.

See [preprocessing_flow.md](preprocessing_flow.md) for the function call graph and feature definitions.
