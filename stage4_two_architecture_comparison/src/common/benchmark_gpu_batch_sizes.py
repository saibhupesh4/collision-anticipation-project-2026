"""Benchmark sequential-video YOLO inference batch sizes under a VRAM reserve cap."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_CANDIDATES = [4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256]
VIDEO_IDS = "00822,01924"  # one positive and one negative, fixed across all candidates


def sample_gpu() -> tuple[float, float, float] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        values = [float(x.strip()) for x in result.stdout.splitlines()[0].split(",")]
        return values[0], values[1], values[2]
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--candidates", default=",".join(map(str, DEFAULT_CANDIDATES)))
    parser.add_argument("--video-ids", default=VIDEO_IDS)
    parser.add_argument("--headroom-mb", type=int, default=800)
    args = parser.parse_args()
    root = args.project_root.resolve()
    pipeline = root / "stage4_two_architecture_comparison" / "src" / "common" / "full_dataset_preprocessing.py"
    metadata = root / "data" / "metadata"
    log_dir = metadata / "gpu_batch_benchmark_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metadata / "full_preprocessing_summary.csv"
    candidates = [int(x) for x in args.candidates.split(",") if x.strip()]
    outcomes = []
    for batch in candidates:
        log_path = log_dir / f"batch_{batch}.log"
        command = [
            sys.executable, str(pipeline), "--project-root", str(root), "--video-ids", args.video_ids,
            "--batch-size", str(batch), "--device", "0", "--gpu-headroom-mb", str(args.headroom_mb),
        ]
        samples: list[tuple[float, float, float]] = []
        start = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                value = sample_gpu()
                if value is not None:
                    samples.append(value)
                time.sleep(1.0)
            return_code = proc.returncode
        wall = time.perf_counter() - start
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if not summary_path.is_file():
            break
        summary = pd.read_csv(summary_path, dtype={"video_id": str})
        ids = [v.strip() for v in args.video_ids.split(",") if v.strip()]
        rows = summary[summary.video_id.isin(ids) & (summary.inference_batch_size == batch)]
        expected = int(rows.expected_frames.sum()) if len(rows) == len(ids) else 0
        processed = int(rows.processed_frames.sum()) if len(rows) == len(ids) else 0
        failed = int((rows.status != "completed").sum()) if len(rows) == len(ids) else len(ids)
        error_text = "; ".join(str(x) for x in rows.error.fillna("").tolist()) if len(rows) else ""
        peak_reserved = int(rows.peak_gpu_reserved_bytes.max()) if len(rows) == len(ids) and rows.peak_gpu_reserved_bytes.notna().any() else 0
        peak_allocated = int(rows.peak_gpu_memory_bytes.max()) if len(rows) == len(ids) and rows.peak_gpu_memory_bytes.notna().any() else 0
        max_total_used = max((s[1] for s in samples), default=0.0)
        gpu_util_mean = sum(s[0] for s in samples) / len(samples) if samples else 0.0
        gpu_util_max = max((s[0] for s in samples), default=0.0)
        total_mib = max((s[2] for s in samples), default=4094.0)
        combined_text = log_text + "\n" + error_text
        oom_free = re.search(r"of which ([\d.]+) MiB is free", combined_text, re.IGNORECASE)
        minimum_reported_free_mib = float(oom_free.group(1)) if oom_free else None
        reserve_ok = (total_mib - max_total_used >= args.headroom_mb) and (minimum_reported_free_mib is None or minimum_reported_free_mib >= args.headroom_mb) and failed == 0
        failure_reason = "CUDA out of memory" if "out of memory" in combined_text.lower() else ("processing or frame validation failure" if failed else "")
        outcomes.append({
            "batch_size": batch, "video_ids": args.video_ids, "status": "completed" if return_code == 0 and failed == 0 and processed == expected else "failed",
            "wall_seconds": round(wall, 3), "frames": processed, "expected_frames": expected,
            "end_to_end_frames_per_second": round(processed / wall, 3) if wall else 0.0,
            "gpu_inference_seconds": round(float(rows.inference_seconds.sum()), 3) if len(rows) == len(ids) else 0.0,
            "gpu_inference_frames_per_second": round(processed / float(rows.inference_seconds.sum()), 3) if len(rows) == len(ids) and rows.inference_seconds.sum() > 0 else 0.0,
            "mean_gpu_utilization_percent": round(gpu_util_mean, 2), "max_gpu_utilization_percent": round(gpu_util_max, 2),
            "max_nvidia_smi_memory_mib": max_total_used, "gpu_total_mib": total_mib,
            "peak_torch_allocated_bytes": peak_allocated, "peak_torch_reserved_bytes": peak_reserved,
            "headroom_mib_at_sampled_peak": round(total_mib - max_total_used, 2), "headroom_ok": reserve_ok,
            "minimum_cuda_free_reported_mib": minimum_reported_free_mib,
            "detection_rows": int(rows.detection_rows.sum()) if len(rows) == len(ids) else 0,
            "tracked_rows": int(rows.tracked_rows.sum()) if len(rows) == len(ids) else 0,
            "failure_reason": failure_reason,
            "failed_videos": failed, "log_path": str(log_path),
        })
        write = pd.DataFrame(outcomes)
        write.to_csv(metadata / "gpu_batch_size_benchmark.csv", index=False)
        (metadata / "gpu_batch_size_benchmark.json").write_text(json.dumps({
            "created_utc": datetime.now(timezone.utc).isoformat(), "candidates": candidates,
            "video_ids": args.video_ids, "headroom_target_mib": args.headroom_mb,
            "precision": "FP16 on CUDA", "device": "CUDA:0", "results": outcomes,
        }, indent=2), encoding="utf-8")
        print(f"batch={batch} status={outcomes[-1]['status']} fps={outcomes[-1]['end_to_end_frames_per_second']} peak_vram_mib={max_total_used} headroom_ok={reserve_ok}", flush=True)
        if outcomes[-1]["status"] != "completed" or not reserve_ok:
            print("Stopping at first failed or headroom-violating batch size.", flush=True)
            break
    safe = [r for r in outcomes if r["status"] == "completed" and r["headroom_ok"]]
    best = max(safe, key=lambda x: x["end_to_end_frames_per_second"]) if safe else None
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "candidates": candidates,
        "video_ids": args.video_ids, "headroom_target_mib": args.headroom_mb,
        "optimal_safe_batch_size": best["batch_size"] if best else None,
        "optimal_safe_end_to_end_frames_per_second": best["end_to_end_frames_per_second"] if best else None,
        "estimated_full_dataset_minutes": round(1701166 / best["end_to_end_frames_per_second"] / 60, 1) if best and best["end_to_end_frames_per_second"] else None,
        "results": outcomes,
    }
    (metadata / "gpu_batch_size_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if best:
        print(f"optimal_safe_batch_size={best['batch_size']} estimated_full_dataset_minutes={payload['estimated_full_dataset_minutes']}")


if __name__ == "__main__":
    main()
