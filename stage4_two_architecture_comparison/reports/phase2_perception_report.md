# Phase 2 — Shared Detection and Tracking Smoke Test

**Corrected run:** `run_20261101_015729_shared_5fps`  
**Dataset subset:** all 80 local smoke-test clips (40 positive, 40 negative)  
**Run status:** 80/80 videos completed successfully

## Configuration

- Detector: pretrained Ultralytics YOLOv8n, 3,157,200 parameters; 6,549,796-byte checkpoint.
- Checkpoint: `models/shared/yolov8n.pt`; SHA-256 `F59B3D833E2FF32E194B5BB8E08D211DC7C5BDF144B90D2C8412C47CCFC83B36`.
- Input size: 640; confidence threshold: 0.25; inference precision: FP32; device: CUDA on NVIDIA RTX 3050 A Laptop GPU.
- COCO classes: person, bicycle, car, motorcycle, bus, truck.
- Tracker: Ultralytics ByteTrack defaults, with `track_buffer=30` at 5 FPS (six seconds of lost-track retention). Exact YAML is saved with the run.
- Frame sampling: first decoded frame at or after each clip-relative 0.2-second grid point. Actual elapsed timestamp is saved for every sample.

## Sampling-rate check

Two representative videos (one positive, one negative) were processed at both 5 FPS and 10 FPS. The 10 FPS run used a 60-frame buffer so the lost-track retention remained six seconds. 5 FPS produced 394 observations; 10 FPS produced 786. The class-scoped median track spans were 1.2 and 2.0 seconds at 5 FPS for the negative and positive clips, compared with 0.65 and 1.0 seconds at 10 FPS. These two clips are too few to establish a generally optimal rate. We retained 5 FPS for the full smoke pass because the tracks were longer and the per-frame workload was lower; short fast interactions may still require denser sampling.

The first full pass was preserved as exploratory but not used downstream: its sampling schedule accumulated drift, with some clips sampling at only 4.31 FPS. The corrected clip-relative grid was verified on the same pair and then used for all 80 videos.

## Full 80-video results

| Measure | Observed |
|---|---:|
| Videos processed | 80/80 |
| Sampled frames | 15,051 |
| Effective sample rate | mean 5.013 FPS; range 5.000–5.026 FPS |
| Detection rows | 40,124 |
| Tracked rows | 40,123 |
| Exported video-scoped tracks | 3,006 |
| Processing time (sum of per-video wall times) | 529.3 seconds |
| Model-call time | 317.1 seconds |
| Parquet cache size | 5,098,024 bytes |
| Peak PyTorch CUDA allocation / reservation | 25,854,464 / 41,943,040 bytes |

Class detection rows: car 33,911; person 2,323; truck 3,229; bus 553; bicycle 106; motorcycle 2. These are detector output counts, not ground-truth accuracy measurements. `nvidia-smi` showed about 117 MiB GPU memory in a runtime snapshot; the per-run PyTorch allocator high-water values are recorded above. No live real-time performance claim is made.

## Tracking validation and limitations

The initial raw ByteTrack IDs were not class-safe: in the corrected sampling run, 286 of 2,674 video-local raw IDs (10.7%) appeared with more than one class across 57 videos. The saved cache therefore preserves `bytetrack_id` for audit and exports a class-scoped `track_id` (`class_id:bytetrack_id`). This prevents downstream features from merging different classes. It can fragment a physical object's trajectory when the detector changes its class; the detector's class errors themselves remain.

The corrected cache was checked across all videos for video-ID isolation, duplicate `(frame, track)` rows, and class changes within an exported track ID. No integrity issues were found. Two annotated videos were visually inspected at 5 and 10 FPS: positive clip `00822` and low-light negative clip `01699`. Review found occasional apparent person detections on vehicle windows/reflections and some small/distant objects that are hard to detect. The dataset has no object-level boxes or track labels, so these are qualitative observations only; detection precision/recall and ID-switch accuracy cannot be measured.

One preview at 19.6 seconds in `00822` was initially considered near-event, then checked against metadata: its event is at 19.5 seconds, so the preview is post-event and was excluded from anticipation review. The pre-event frames were checked separately. This correction is preserved in the experiment notes; no post-event frame is used for temporal target generation.

## Gate 2 decision

The shared pipeline now produces reusable, time-stamped caches for all 80 smoke-test videos, with video and class namespaces validated. It is suitable for integration testing. The detector's qualitative limits and lack of object-level ground truth must be carried into all later reports. The smoke subset cannot support claims about detector performance or collision anticipation generalization.

Run outputs and review clips:

- Corrected cache: `experiments/runs/run_20261101_015729_shared_5fps/`
- Annotated positive/negative videos: `outputs/videos/perception_validation/run_20261101_015729_shared_5fps/`
- Reproducible runner: `src/common/run_shared_perception.py`
- Cached-annotation utility: `src/common/annotate_cached_tracks.py`
