# Phase 0 Repository and Local Data Audit

**Date:** 2026-11-01

## Workspace inventory

The workspace root has no project code or documentation. Its only visible entry is `data/`, containing:

- `metadata/positive_metadata.csv` — 750 records.
- `metadata/negative_metadata.csv` — 750 records.
- `metadata/stage3_sample_manifest.csv` — 80 records, seed stated in the execution prompt as 42.
- `metadata/video_properties.csv` — 80 records.
- `raw/positive/` — 40 MP4 files.
- `raw/negative/` — 40 MP4 files.

No source files, notebooks, prior stage implementations, previous experiment reports, metrics, checkpoints, or local Git repository were found. This statement applies to this workspace only; no other project location was searched.

## Local sample integrity

The manifest labels are encoded as `1` (positive) and `0` (negative). After mapping these to their respective folders, all 80 manifest filenames matched exactly one local file; there were no missing or unmanifested MP4s. OpenCV opened all 80 files and reported positive frame counts. Resolution was 1280×720 throughout. Reported FPS ranged from 28.9 to 31.0; frame counts ranged from 540 to 1,297. Total MP4 size was 1,327,959,890 bytes (about 1.24 GiB).

This verifies basic readability only. It does not revalidate temporal metadata, annotation semantics, visual detection/tracking quality, or benchmark suitability. The 80-video subset remains a pipeline smoke-test sample, not a final benchmark.

## Metadata and known risks

Both metadata tables have 750 records and the columns `file_name`, `time_of_event`, `time_of_alert`, `light_conditions`, `weather`, `scene`, and `time_to_accident`. Positive rows have populated event and alert timestamps; negative rows have neither. `time_to_accident` is empty in every row. Each class has one missing weather value.

Scene composition differs: positives include 475 Urban and 171 Highway records; negatives include 309 Urban and 208 Highway records, with different counts in other categories as well. This is a potential shortcut risk. Scene, weather, lighting, filename, and IDs must not be model inputs in the primary experiment.

The meanings and timestamp conventions for `time_of_event` and `time_of_alert` have not yet been independently verified against official documentation. They must not be used to define labels until Phase 1 records their verified semantics and any ambiguity.

## Remote data access

The initial S3 request, under the restricted network context, could not connect. A read-only retry through the approved network path with the existing AWS CLI `hf` profile succeeded. The bucket contains `train/`, `test-public/`, and `test-private/` prefixes plus README and license files. No credentials were printed or modified. No video objects from the bucket were downloaded.

## Phase 0 conclusion

The local smoke-test files are present and readable. This workspace does not contain prior pipeline code or experiment outputs to reuse. Phase 1 must first verify timestamp semantics, inspect the official dataset documentation and licence, and record the dataset limitations before any label construction or training.
