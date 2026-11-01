"""Train static and trajectory XGBoost smoke models on video-level splits."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score
from xgboost import XGBClassifier


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def score_split(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int | str]:
    if len(y) == 0 or len(np.unique(y)) < 2:
        ap = float("nan")
    else:
        ap = float(average_precision_score(y, scores))
    pred = (scores >= threshold).astype(int)
    return {
        "examples": int(len(y)),
        "positive_examples": int(y.sum()),
        "negative_examples": int(len(y) - y.sum()),
        "average_precision": ap,
        "precision_at_validation_threshold": float(precision_score(y, pred, zero_division=0)) if len(y) else float("nan"),
        "recall_at_validation_threshold": float(recall_score(y, pred, zero_division=0)) if len(y) else float("nan"),
        "f1_at_validation_threshold": float(f1_score(y, pred, zero_division=0)) if len(y) else float("nan"),
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
    if data.empty:
        raise ValueError("Feature file is empty")
    feature_sets = {
        "static_xgboost": [c for c in data.columns if c.startswith("static__")],
        "trajectory_xgboost": [c for c in data.columns if c.startswith("trajectory__")],
    }
    if not feature_sets["static_xgboost"] or not feature_sets["trajectory_xgboost"]:
        raise ValueError("Static or trajectory features missing")

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_architecture1_xgboost_smoke"
    root = Path.cwd() / "stage4_two_architecture_comparison"
    run_dir = root / "experiments" / "runs" / run_id
    model_dir = root / "models" / "architecture_1" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "run_id": run_id,
        "architecture": "Architecture 1",
        "dataset_subset": "400-video preprocessing subset",
        "features": str(feature_path),
        "split_manifest": str(root / "data" / "manifests" / "subset_400_manifest.csv"),
        "seed": args.seed,
        "horizons_sec": sorted(float(x) for x in data.horizon_sec.unique()),
        "feature_sets": {k: v for k, v in feature_sets.items()},
        "xgboost": {"n_estimators": 80, "max_depth": 2, "learning_rate": 0.04, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0, "n_jobs": 2},
        "training_scope": "400-video pipeline-validation subset; not a final research estimate",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics = []
    prediction_rows = []
    for horizon in config["horizons_sec"]:
        subset = data[data.horizon_sec == horizon].copy()
        train, validation, test = (subset[subset.split == split].copy() for split in ("train", "validation", "test"))
        for model_name, columns in feature_sets.items():
            x_train = train[columns].to_numpy(dtype=np.float32)
            y_train = train.label.to_numpy(dtype=int)
            if len(np.unique(y_train)) < 2:
                raise ValueError(f"Training split has one class at horizon {horizon}")
            model = XGBClassifier(
                n_estimators=80,
                max_depth=2,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=2,
                random_state=args.seed,
            )
            start_time = time.perf_counter()
            model.fit(x_train, y_train, verbose=False)
            training_seconds = time.perf_counter() - start_time
            v_scores = model.predict_proba(validation[columns].to_numpy(dtype=np.float32))[:, 1]
            threshold = choose_threshold(validation.label.to_numpy(dtype=int), v_scores)
            test_scores = model.predict_proba(test[columns].to_numpy(dtype=np.float32))[:, 1]
            test_metrics = score_split(test.label.to_numpy(dtype=int), test_scores, threshold)
            checkpoint = model_dir / f"{model_name}_h{str(horizon).replace('.', '_')}.json"
            model.save_model(checkpoint)
            metric = {
                "model": model_name,
                "horizon_sec": horizon,
                "training_seconds": training_seconds,
                "train_examples": int(len(train)),
                "validation_examples": int(len(validation)),
                "validation_threshold": threshold,
                "test_metrics": test_metrics,
                "boosted_trees": len(model.get_booster().get_dump()),
                "checkpoint": str(checkpoint),
                "scope": "400-video pipeline-validation subset; not a final research estimate; not a generalization estimate",
            }
            metrics.append(metric)
            for split_name, frame, scores in (
                ("train", train, model.predict_proba(train[columns].to_numpy(dtype=np.float32))[:, 1]),
                ("validation", validation, v_scores),
                ("test", test, test_scores),
            ):
                for idx, (_, sample) in enumerate(frame.iterrows()):
                    prediction_rows.append({
                        "model": model_name,
                        "horizon_sec": horizon,
                        "video_id": sample.video_id,
                        "split": split_name,
                        "label": int(sample.label),
                        "score": float(scores[idx]),
                        "endpoint_sec": float(sample.endpoint_sec),
                        "threshold_from_validation": threshold,
                    })

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(run_dir / "predictions.csv", index=False)
    with (run_dir / "experiment_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["run_id", "architecture", "model", "dataset_subset", "split", "configuration", "seed", "completion_status", "checkpoint_location", "report_location"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"run_id": run_id, "architecture": "Architecture 1", "model": "XGBoost static baseline + trajectory", "dataset_subset": "400-video preprocessing subset", "split": "inherited video-level split; 400-video subset", "configuration": str(run_dir / "config.json"), "seed": args.seed, "completion_status": "completed", "checkpoint_location": str(model_dir), "report_location": str(run_dir / "metrics.json")})
    print(f"run_id={run_id} models={len(metrics)} path={run_dir}")
    print(pd.DataFrame([{"model": x["model"], "horizon_sec": x["horizon_sec"], "test_ap_smoke": x["test_metrics"]["average_precision"], "test_n": x["test_metrics"]["examples"]} for x in metrics]).to_string(index=False))


if __name__ == "__main__":
    main()



