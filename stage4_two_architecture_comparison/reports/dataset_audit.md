# Dataset Audit

**Audit date:** 2026-11-01  
**Local source:** `data/metadata/` and `data/raw/` in the workspace  
**Published source:** Nexar Collision Prediction Dataset, Hugging Face repository `nexar-ai/nexar_collision_prediction`; Kaggle challenge documentation; CVPRW 2025 paper.

## Dataset structure and access

The published training set contains 1,500 labeled videos, balanced as 750 positive and 750 negative, typically about 40 seconds at 1280×720 and approximately 30 FPS. Positive labels include both collisions and near misses; negatives represent regular driving. The challenge also provided 1,344 approximately 10-second test clips split into public and private evaluation subsets. Those test clips are not locally present and are not used here.

The available workspace has all 1,500 metadata rows but only the pre-existing 80-video sample (40 per class). All 80 sample files are readable and map exactly to the sample manifest. They occupy 1,327,959,890 bytes. Their reported dimensions are 1280×720, FPS spans 28.9–31.0, and frame counts span 540–1,297. The observed sample timestamp values are within each sampled video's recorded duration.

The current public Hugging Face repository revision observed on 2026-11-01 is `7535d0656dac31d7da2846913bb69c6331d2a70a` (last modified 2026-09-29). The user's bucket does not contain a source revision marker, so the exact commit copied there cannot be independently pinned. Its root README, license, train filename set, and 1,500-row local metadata are consistent with the public training dataset. A read-only list retry succeeded using the existing `hf` AWS profile; the earlier failure was network-context related, not an authentication failure.

Bucket inventory: `train/` has 1,502 objects totaling 25,529,607,131 bytes, including exactly 1,500 MP4s totaling 25,529,548,265 bytes. All 1,500 remote MP4 names match the local labeled metadata. The 80 local MP4s match remote names; the remaining 1,420 training videos total 24,201,588,375 bytes (about 24.20 GB or 22.54 GiB). C: had 291,611,488,256 bytes free at audit time, leaving about 267.4 GB after a full training-video download before any processing outputs. The `test-public/` prefix contains 669 objects (2,881,274,944 bytes) and `test-private/` contains 679 objects (2,968,074,170 bytes); neither test prefix was downloaded or used. Transfer duration has not been measured, so no time estimate is claimed. Selective per-object downloads are possible through the configured S3 CLI.

The bucket's 2,520-byte `LICENSE` was copied into [NEXAR_LICENSE.txt](NEXAR_LICENSE.txt). It requires Nexar attribution and restricts resale, harmful-system use, manipulated media, reidentification/surveillance, weaponization, and exploitative practices. The Hugging Face card additionally gates access on consent to sharing contact information. No video data were downloaded in this phase.

The Hugging Face card identifies the Nexar Open Data License, requires attribution, and prohibits resale, reidentification, weapons development, and harmful use. It also states that access requires accepting conditions and sharing contact information. The local workspace does not contain the license text or proof of the acceptance event; retain and follow the license attached to the original acquisition. Do not redistribute raw videos. The source information is available on the [official Hugging Face dataset card](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction).

## Timestamp semantics and target use

Official challenge documentation describes both timestamp fields as elapsed seconds in the video:

- `time_of_event` marks the collision or near-miss event time. It is populated for positive training examples and is the timestamp used to define future-event horizons. For near misses, this is an annotated near-collision event, not necessarily physical contact.
- `time_of_alert` marks the annotation's earliest anticipation point. The paper defines it as the earliest moment when a focused human could recognize that a dangerous situation was about to occur; it is based on annotations from multiple annotators and is explicitly subjective. It is a human anticipation reference, not a real-time warning actually emitted to a driver and not an objective physical ground truth. Use it only for label interpretation and secondary analysis; never as a model input.
- `time_to_accident` is blank in every row of the local 1,500-row training metadata. The Hugging Face card explains that the corresponding test field describes how far before the event a test clip was cut and is unavailable in the untrimmed training set. Keep the raw field blank; do not substitute `time_of_event - time_of_alert` and call that quantity `time_to_accident`.

The Kaggle documentation uses seconds elapsed in the clip for both event and alert timestamps. The official card and paper support their role and ordering. The exact frame-rounding convention (for example, whether the timestamp denotes a frame start, center, or inclusive boundary) is not specified in the accessible documentation and remains unresolved. Code must preserve timestamps as seconds and avoid claiming frame-level precision beyond the source annotations.

Official reference points: [Kaggle challenge data description](https://www.kaggle.com/competitions/nexar-collision-prediction/data), [CVPRW 2025 paper](https://openaccess.thecvf.com/content/CVPR2025W/WAD/papers/Moura_Nexar_Dashcam_Collision_Prediction_Dataset_and_Challenge_CVPRW_2025_paper.pdf), and [Hugging Face dataset card](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction).

## Metadata audit and bias

Each class has 750 rows. Positive examples have both timestamps populated; negative examples have neither. `time_to_accident` is empty for all 1,500 records. One weather value is missing in each class. No positive row has a missing/nonpositive event or alert timestamp, and no row has an alert later than its event. Across positive records, event time ranges from 3.032 to 56.8 seconds and event-minus-alert ranges from about 0.033 to 4.466 seconds (median about 1.433 seconds).

Scene labels differ by class: positives contain 475 Urban and 171 Highway rows; negatives contain 309 Urban and 208 Highway rows. This is a shortcut risk. The primary models must not use scene, weather, lighting, filenames, IDs, or other metadata. Environmental metadata may be used only for descriptive audits and error analysis, with small slices clearly disclosed.

The 80-video sample is not representative enough for generalization claims. It is a pipeline smoke-test subset only. Its scene mix is also class-dependent: the 40 positives contain 26 Urban and 9 Highway clips, while the 40 negatives contain 16 Urban and 11 Highway clips (with the remainder spread across other categories). This sample-level difference further limits interpretation of smoke-test behavior.

## Labels and geometry limitations

The dataset provides video-level collision/near-miss labels and event/alert times. It does not supply object-level collision labels, ground-truth pedestrian/vehicle trajectories, verified metric distances, camera calibration, or complete ego motion. The available sample alone does not establish whether each positive event threatens the dashcam vehicle. Architecture 2 must distinguish ego-path threats from third-party events and must not claim calibrated physical distance or time-to-collision from image-space measurements.

The generated [event relevance review manifest](../data/manifests/event_relevance_review.csv) leaves all positive relevance labels as pending human review (`Unclear` is a placeholder only). No scene-level inference has been treated as confirmed truth.

## Split and leakage plan

The generated split manifest contains all 1,500 labeled training videos with a deterministic seed-42, class-stratified, video-level 70/15/15 proposal: 1,050 train, 224 validation, and 226 test videos. Per class the counts are 525/112/113. This is a proposed split, not an official split or a measured benchmark. No source-recording/session group IDs were found; if that information becomes available, related clips must be grouped before finalizing the split. Only 80 videos are local, so the other 1,420 rows are metadata-only and cannot be processed yet.

Generate temporal windows only after assigning each video to one split. For a prediction horizon `h`, define the target at observation time `t` from whether an annotated positive event occurs in `(t, t+h]`. Exclude observation windows that contain any post-event frames. Negative videos have no event timestamp and remain negative. Store `time_of_alert` separately as the human-reference marker. Fit preprocessing and thresholds on train/validation only; keep the proposed test videos untouched until final evaluation. The implementation and window rules are documented in [window generation specification](window_generation_spec.md).

## Current blockers and next steps

1. Resolve endpoint reachability before accessing any additional files; do not re-download the existing 80 videos.
2. Preserve license and access conditions from the original dataset acquisition.
3. Keep the split provisional until any available session-group information is checked.
4. Complete human event-relevance review before interpreting Architecture 2 as ego-centric evaluation.
5. Continue to use the 80 local videos only for pipeline validation, not research performance claims.
