# Expanded Pilot Download Proposal — Approval Gate

**Prepared:** 2026-11-01  
**State:** No transfer started. Awaiting user approval.

## Proposed selection

The 80 existing clips count toward the requested 50 positive / 50 negative pilot. The following 20 missing training clips were selected deterministically (seed 42) from the remaining videos, ten per class, and assigned the split already recorded in the provisional manifest:

| Class | Video IDs | Proposed split counts |
|---|---|---|
| Positive (10) | 00104, 00131, 00179, 00236, 00434, 00455, 00528, 00558, 00798, 00891 | train 7, test 3 |
| Negative (10) | 01175, 01250, 01532, 01605, 01647, 01763, 01817, 01931, 02013, 02079 | train 6, validation 2, test 2 |

These IDs are all absent locally and were confirmed in the bucket listing. The original split is provisional; source-session grouping data is unavailable. No test-private videos are included.

## Resource estimate

- Exact transfer size from the bucket object sizes: **383,841,531 bytes** (366.06 MiB; 0.357 GiB).
- Current C: free space: **291,649,294,336 bytes**. After this transfer, approximately **291,265,452,805 bytes** would remain, before cache/model outputs.
- At a planning throughput of 1–5 MB/s, payload transfer is approximately **1.3–6.4 minutes**. The observed transfer rate is not available, so this is a bandwidth-based estimate; connection setup/retries may add time.
- Total remaining train-video bucket data is 1,420 videos / 24,201,588,375 bytes. This proposal downloads only the 20 listed videos.

The 80-video pipeline smoke tests pass and both architectures run end to end. This proposal expands the pilot to 50 videos per class. **Approval is required before downloading these files.**
