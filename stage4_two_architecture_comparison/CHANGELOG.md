# Changelog

## 2026-11-01

- Added Phase 0 repository, hardware, environment, local-data, and access audits.
- Added Phase 1 dataset/timestamp audit and leakage-aware window specification.
- Added reproducible manifest builder for the 1,500 labeled training metadata rows, including a seed-42 video-level class-stratified split proposal and pending human event-relevance review entries.
- Added the official bucket license copy and recorded bucket inventory and current Hugging Face revision.
- Ran the shared YOLOv8n/ByteTrack pipeline on all 80 local smoke-test videos, corrected sampling-grid drift and class-crossing raw IDs, and saved 5 FPS detection/track caches plus annotated review clips.
- Implemented and smoke-trained Architecture 1 (static/temporal XGBoost and GRU); documented small-test results and shortcut-feature risk.
- Implemented Architecture 2 (Kalman prediction, masked optical-flow camera motion, image-space ego corridor/risk fusion) and completed full 80-video camera extraction, rule/fusion/no-camera smoke runs, and synthetic tests.
- Passed all 10 Architecture 1/2 unit tests and refreshed the 20-run experiment registry.
- Preserved exploratory/failed run outputs and documented perception and model limitations. No dataset videos were downloaded beyond the existing 80.
- Prepared a deterministic 20-video expanded-pilot selection with exact storage estimate; transfer remains pending user approval.
- Added resumable full-frame YOLOv8n/ByteTrack preprocessing and image-space trajectory/interaction feature extraction with per-frame indexing and validation.
- Benchmarked CUDA FP16 batch sizes 4–256; batch 16 was fastest safe on 2,414 fixed smoke frames at 94.1 end-to-end frames/sec. Batch 256 failed the VRAM headroom constraint.
- User redirected the run from all 1,500 videos to a deterministic 200-positive/200-negative subset. The partial all-data cache and benchmark scratch outputs were cleaned for handoff; source videos, source metadata, prior Stage 4 reports, models, and experiments were preserved.
