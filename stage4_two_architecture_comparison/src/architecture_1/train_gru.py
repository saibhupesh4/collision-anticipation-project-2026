"""Train a compact GRU on short pre-event trajectory feature sequences."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
from torch import nn


class TinyGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 16):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        sequence, _ = self.gru(x)
        return self.head(sequence[:, -1]).squeeze(-1)


def threshold_from_validation(y: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True, help="Architecture 1 feature run directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    feature_dir = args.features if args.features.is_absolute() else Path.cwd() / args.features
    archive = np.load(feature_dir / "sequences.npz", allow_pickle=False)
    sequences = archive["sequences"].astype(np.float32)
    metadata = pd.read_csv(feature_dir / "sequence_metadata.csv", dtype={"video_id": str})
    feature_names = json.loads((feature_dir / "gru_feature_names.json").read_text(encoding="utf-8"))
    if sequences.ndim != 3 or sequences.shape[0] != len(metadata) or sequences.shape[2] != len(feature_names):
        raise ValueError("Sequence data and metadata do not align")

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_architecture1_gru_smoke"
    root = Path.cwd() / "stage4_two_architecture_comparison"
    run_dir = root / "experiments" / "runs" / run_id
    model_dir = root / "models" / "architecture_1" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = {
        "run_id": run_id,
        "architecture": "Architecture 1 GRU",
        "dataset_subset": "400-video preprocessing subset",
        "features": str(feature_dir),
        "seed": args.seed,
        "device": str(device),
        "sequence_shape": list(sequences.shape[1:]),
        "feature_names": feature_names,
        "hidden_size": 16,
        "maximum_epochs": args.epochs,
        "batch_size": 16,
        "early_stopping_patience": 6,
        "scope": "400-video pipeline-validation subset; not a final research estimate; not a generalization estimate",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    all_predictions = []
    metrics = []
    histories = []
    for horizon in sorted(float(x) for x in metadata.horizon_sec.unique()):
        horizon_mask = metadata.horizon_sec.to_numpy(dtype=float) == horizon
        frame = metadata[horizon_mask].reset_index(drop=True)
        data = sequences[horizon_mask]
        indices = {name: np.flatnonzero(frame.split.to_numpy() == name) for name in ("train", "validation", "test")}
        y = {name: frame.iloc[idx].label.to_numpy(dtype=np.float32) for name, idx in indices.items()}
        if len(np.unique(y["train"])) < 2:
            raise ValueError(f"Training split has one class at horizon {horizon}")

        # Fit sequence normalization on training videos only.
        train_values = data[indices["train"]].reshape(-1, data.shape[-1])
        mean = train_values.mean(axis=0)
        scale = train_values.std(axis=0)
        scale[scale < 1e-6] = 1.0
        normalized = (data - mean[None, None, :]) / scale[None, None, :]
        xs = {name: torch.tensor(normalized[idx], dtype=torch.float32, device=device) for name, idx in indices.items()}
        ys = {name: torch.tensor(y[name], dtype=torch.float32, device=device) for name in indices}

        model = TinyGRU(data.shape[-1], 16).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
        criterion = nn.BCEWithLogitsLoss()
        best_state = None
        best_val_loss = float("inf")
        patience = 0
        history = []
        started = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            permutation = torch.randperm(len(xs["train"]), device=device)
            train_losses = []
            for batch_start in range(0, len(permutation), 16):
                batch = permutation[batch_start:batch_start + 16]
                optimizer.zero_grad(set_to_none=True)
                logits = model(xs["train"][batch])
                loss = criterion(logits, ys["train"][batch])
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                val_loss = float(criterion(model(xs["validation"]), ys["validation"]).cpu())
            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            history.append({"horizon_sec": horizon, "epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 6:
                    break
        training_seconds = time.perf_counter() - started
        if best_state is None:
            raise RuntimeError("GRU training did not produce a finite validation checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(xs["validation"])).cpu().numpy()
            test_scores = torch.sigmoid(model(xs["test"])).cpu().numpy()
        threshold = threshold_from_validation(y["validation"], val_scores)
        test_pred = (test_scores >= threshold).astype(int)
        test_metrics = {
            "examples": int(len(y["test"])),
            "positive_examples": int(y["test"].sum()),
            "negative_examples": int(len(y["test"]) - y["test"].sum()),
            "average_precision": float(average_precision_score(y["test"], test_scores)) if len(np.unique(y["test"])) > 1 else float("nan"),
            "precision_at_validation_threshold": float(precision_score(y["test"], test_pred, zero_division=0)),
            "recall_at_validation_threshold": float(recall_score(y["test"], test_pred, zero_division=0)),
            "f1_at_validation_threshold": float(f1_score(y["test"], test_pred, zero_division=0)),
            "scope": "400-video pipeline-validation subset; not a final research estimate",
        }
        checkpoint = model_dir / f"gru_h{str(horizon).replace('.', '_')}.pt"
        torch.save({"state_dict": best_state, "mean": mean, "scale": scale, "feature_names": feature_names, "horizon_sec": horizon}, checkpoint)
        metrics.append({"model": "gru", "horizon_sec": horizon, "training_seconds": training_seconds, "epochs_run": len(history), "validation_threshold": threshold, "test_metrics": test_metrics, "parameters": sum(p.numel() for p in model.parameters()), "checkpoint": str(checkpoint)})
        histories.extend(history)
        for split_name in ("train", "validation", "test"):
            with torch.no_grad():
                scores = torch.sigmoid(model(xs[split_name])).cpu().numpy()
            for local_i, global_i in enumerate(indices[split_name]):
                row = frame.iloc[global_i]
                all_predictions.append({"model": "gru", "horizon_sec": horizon, "video_id": row.video_id, "split": split_name, "label": int(row.label), "score": float(scores[local_i]), "endpoint_sec": float(row.endpoint_sec), "threshold_from_validation": threshold})

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    pd.DataFrame(all_predictions).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(histories).to_csv(run_dir / "training_history.csv", index=False)
    with (run_dir / "experiment_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["run_id", "architecture", "model", "dataset_subset", "split", "configuration", "seed", "completion_status", "checkpoint_location", "report_location"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"run_id": run_id, "architecture": "Architecture 1", "model": "GRU trajectory", "dataset_subset": "400-video preprocessing subset", "split": "inherited video-level split; 400-video subset", "configuration": str(run_dir / "config.json"), "seed": args.seed, "completion_status": "completed", "checkpoint_location": str(model_dir), "report_location": str(run_dir / "metrics.json")})
    print(f"run_id={run_id} models={len(metrics)} device={device} path={run_dir}")
    print(pd.DataFrame([{"horizon_sec": x["horizon_sec"], "test_ap_smoke": x["test_metrics"]["average_precision"], "test_n": x["test_metrics"]["examples"], "epochs": x["epochs_run"]} for x in metrics]).to_string(index=False))


if __name__ == "__main__":
    main()

