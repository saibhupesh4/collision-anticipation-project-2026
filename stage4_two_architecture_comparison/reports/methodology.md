# Methodology

## Dataset and temporal targets

The experiment uses Nexar's labeled training data only. Video IDs define the split boundary. A deterministic seed-42, class-stratified 70/15/15 video split is recorded in `data/manifests/split_manifest.csv`; it is provisional because source-session grouping is unavailable. Temporal windows are created only after splitting. For horizon `h`, an endpoint `t` is positive when an annotated event lies in `(t, t+h]`; windows containing post-event frames are excluded. The human-derived alert timestamp remains a separate analysis reference and is never a model input. Full rules are in [window_generation_spec.md](window_generation_spec.md).

## Shared perception

Both architectures receive the same per-video YOLO detections and ByteTrack tracks. The current implementation uses pretrained YOLOv8n, 640×640 inference, confidence 0.25, and COCO classes person, bicycle, car, motorcycle, bus, and truck. Inference is sequential on CUDA, full precision, with small frame batches (one frame at a time). No detector fine-tuning is performed.

Sampling selects the first decoded frame at or after each clip-relative `k / sample_rate` target, using OpenCV's elapsed frame timestamp; frame-index/FPS fallback is used only if that timestamp is invalid. The initial 5 FPS setting is retained for smoke processing after a two-video 5/10 FPS comparison: 10 FPS doubles observations but yielded shorter median tracks in those clips. This comparison is too small to establish a generally optimal sampling rate.

Ultralytics ByteTrack defaults are recorded per run. The lost-track buffer is scaled to maintain a six-second time window as sampling rate changes. Raw ByteTrack IDs are retained in `bytetrack_id`; exported `track_id` is namespaced as `class_id:bytetrack_id` because raw ByteTrack IDs crossed object classes in the initial pass. This prevents downstream feature code from merging classes, while class flicker can still fragment a physical object's trajectory. Each row also carries the video ID, class, box, confidence, elapsed timestamp, track age, and time since the previous observation.

Structured per-video detections and tracks are saved as Parquet under each unique experiment run. Run config, per-video summaries, class counts, and a registry entry are saved alongside them. Two positive/negative videos were annotated from cache at 5 FPS and 10 FPS; annotation generation does not rerun the detector.

## Full-frame preprocessing handoff

For the faster user-requested subset run, `src/common/full_dataset_preprocessing.py` uses the same pretrained YOLOv8n checkpoint on CUDA:0 for every decoded frame. It uses batch 16 and FP16, with a PyTorch allocator cap reserving at least 800 MiB plus additional display/driver margin. Ultralytics handles BGR-to-RGB conversion, resize/letterbox, and input normalization. ByteTrack updates each result in source-frame order; videos are processed sequentially. CPU remains responsible for video decoding, tracker bookkeeping, feature calculations, and storage. The dataset itself is not loaded into VRAM.

Batch size was swept from 4 through 256 on two fixed clips (2,414 frames). Batch 16 produced the best observed end-to-end throughput (94.1 frames/sec). Batch 256 failed with CUDA out-of-memory and about 300 MiB free. The selected 200-positive/200-negative subset contains 450,882 frames, approximately 80 minutes at the measured sweep rate. This is a small-sample estimate. Its membership is selected within the original seed-42 split, retaining video-level isolation and 70/15/15 proportions per class.

## Architecture 1

Architecture 1 uses image-space positions, velocities, acceleration proxies, scale changes, track persistence, and pairwise interaction features from the shared tracks. It compares static and trajectory XGBoost with a small GRU. Absolute image-center features are excluded. Pixel coordinates and apparent expansion remain image-space cues; they are not labeled metric distance or physical TTC. Models were smoke-trained; see [architecture_1_report.md](architecture_1_report.md).

## Architecture 2

Architecture 2 reuses the same detections/tracks and adds masked optical-flow camera-motion compensation, timestamp-aware constant-velocity Kalman prediction, prediction uncertainty, a broad image-space ego corridor, conflict overlap, and risk fusion. A rule-only baseline and no-camera ablation are included. No camera calibration, ego-motion ground truth, or metric road geometry is available. Any corridor/risk quantity is a relative image-space approximation, not a calibrated road plane, physical distance, or TTC. Synthetic checks validate implementation logic only; they do not represent real-world Nexar performance. Full smoke results are in [architecture_2_report.md](architecture_2_report.md).

## Fair evaluation and limits

Both architectures use identical provisional video splits and event-relative horizons. Scaling is fit on training data; thresholds are selected on validation data. The current 80-video sample has only 12 test videos, is for integration smoke testing, and cannot support generalization claims or an architecture recommendation. Nexar supplies video-level event labels, not object-level collisions, trajectories, ego involvement, or metric geometry. The event relevance review is still pending human review; no quantitatively validated ego-threat false-positive or missed-threat analysis is possible.
