"""Fit Logistic Regression risk fusion and camera-motion ablation on smoke data."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def validation_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, object]:
    pred = (score >= threshold).astype(int)
    return {
        "examples": int(len(y)), "positive_examples": int(y.sum()), "negative_examples": int(len(y) - y.sum()),
        "average_precision": float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else float("nan"),
        "precision_at_validation_threshold": float(precision_score(y, pred, zero_division=0)),
        "recall_at_validation_threshold": float(recall_score(y, pred, zero_division=0)),
        "f1_at_validation_threshold": float(f1_score(y, pred, zero_division=0)),
        "scope": "400-video pipeline-validation subset; not a final research estimate",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    feature_path = args.features if args.features.is_absolute() else Path.cwd() / args.features
    if feature_path.is_dir():
        feature_path = feature_path / "features.parquet"
    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature parquet file not found: {feature_path}")
    data = pd.read_parquet(feature_path)
    sets = {
        "no_camera_logistic": [c for c in data if c.startswith("no_camera__")],
    }
    if any(not columns for columns in sets.values()):
        raise ValueError("Architecture 2 feature groups are missing")
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_architecture2_logistic_smoke"
    root = Path.cwd() / "stage4_two_architecture_comparison"
    run_dir = root / "experiments" / "runs" / run_id
    model_dir = root / "models" / "architecture_2" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    config = {"run_id": run_id, "architecture": "Architecture 2", "features": str(feature_path), "dataset_subset": "400-video preprocessing subset", "seed": args.seed, "models": sets, "logistic_regression": {"C": 0.5, "class_weight": None, "scaler": "StandardScaler fit on training videos only"}, "geometry": "image_space_heuristic; metric geometry invalid", "scope": "400-video pipeline-validation subset; not a final research estimate"}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    results, predictions = [], []
    for horizon in sorted(float(v) for v in data.horizon_sec.unique()):
        subset = data[data.horizon_sec == horizon].copy()
        train, val, test = [subset[subset.split == s].copy() for s in ("train", "validation", "test")]
        y_train, y_val, y_test = [frame.label.to_numpy(dtype=int) for frame in (train, val, test)]
        # Rule-only score is recorded as a transparent baseline, with no fitting.
        rule_scores = test["comp__rule_risk"].to_numpy(dtype=float)
        rule_metrics = metrics(y_test, rule_scores, 0.5)
        results.append({"model": "rule_only", "horizon_sec": horizon, "training_seconds": 0.0, "validation_threshold": 0.5, "test_metrics": rule_metrics, "features": ["comp__rule_risk"]})
        for name, columns in sets.items():
            model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000, random_state=args.seed))
            start = time.perf_counter()
            model.fit(train[columns].to_numpy(dtype=float), y_train)
            training_seconds = time.perf_counter() - start
            val_scores = model.predict_proba(val[columns].to_numpy(dtype=float))[:, 1]
            threshold = validation_threshold(y_val, val_scores)
            test_scores = model.predict_proba(test[columns].to_numpy(dtype=float))[:, 1]
            checkpoint = model_dir / f"{name}_h{str(horizon).replace('.', '_')}.joblib"
            joblib.dump(model, checkpoint)
            results.append({"model": name, "horizon_sec": horizon, "training_seconds": training_seconds, "validation_threshold": threshold, "test_metrics": metrics(y_test, test_scores, threshold), "features": columns, "checkpoint": str(checkpoint)})
            for split_name, frame, score in (
                ("train", train, model.predict_proba(train[columns].to_numpy(dtype=float))[:, 1]),
                ("validation", val, val_scores),
                ("test", test, test_scores),
            ):
                predictions.extend({"model": name, "horizon_sec": horizon, "video_id": r.video_id, "split": split_name, "label": int(r.label), "score": float(score[i]), "endpoint_sec": float(r.endpoint_sec), "threshold_from_validation": threshold} for i, r in enumerate(frame.itertuples(index=False)))
        predictions.extend({"model": "rule_only", "horizon_sec": horizon, "video_id": r.video_id, "split": "test", "label": int(r.label), "score": float(rule_scores[i]), "endpoint_sec": float(r.endpoint_sec), "threshold_from_validation": 0.5} for i, r in enumerate(test.itertuples(index=False)))

    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=True), encoding="utf-8")
    pd.DataFrame(predictions).to_csv(run_dir / "predictions.csv", index=False)
    with (run_dir / "experiment_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["run_id", "architecture", "model", "dataset_subset", "split", "configuration", "seed", "completion_status", "checkpoint_location", "report_location"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"run_id": run_id, "architecture": "Architecture 2", "model": "rule risk + Logistic Regression fusion + camera-motion ablation", "dataset_subset": "400-video preprocessing subset", "split": "inherited video-level split; 400-video subset", "configuration": str(run_dir / "config.json"), "seed": args.seed, "completion_status": "completed", "checkpoint_location": str(model_dir), "report_location": str(run_dir / "metrics.json")})
    print(f"run_id={run_id} models={len(results)} path={run_dir}")
    print(pd.DataFrame([{"model": r["model"], "horizon_sec": r["horizon_sec"], "test_ap_smoke": r["test_metrics"]["average_precision"], "test_n": r["test_metrics"]["examples"]} for r in results]).to_string(index=False))


if __name__ == "__main__":
    main()


