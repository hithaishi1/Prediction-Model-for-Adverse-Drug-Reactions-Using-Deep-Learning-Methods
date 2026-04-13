"""
Evaluation script for ADR prediction models.

Supports MLP, ResNet, Attention, and EmbeddingMLP. Pass --model to select
which model to evaluate, and --dataset to select the processed data directory.

Calibration
-----------
Pass --calibrate to apply temperature scaling after standard evaluation.
Temperature scaling fits a single scalar T on the validation set by minimizing
negative log-likelihood, then divides the test-set logits by T before applying
sigmoid. T > 1 softens predictions (spreads probabilities toward 0.5); T < 1
sharpens them. This corrects systematic over- or under-confidence without
retraining and does not change AUROC. Calibrated metrics are printed and saved
alongside uncalibrated metrics in the results JSON.

Usage:
    python src/evaluate.py --model mlp
    python src/evaluate.py --model resnet --dataset min1000_expanded
    python src/evaluate.py --model attention --dataset min1000_expanded
    python src/evaluate.py --model embedding --dataset min1000_expanded
    python src/evaluate.py --model all                  # mlp, resnet, attention only
    python src/evaluate.py --model mlp --calibrate      # with temperature scaling
"""

import sys
import os
import argparse
import json
import pickle
from pathlib import Path
from scipy.optimize import minimize_scalar

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report,
)
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from models import get_model, EmbeddingMLP
    from train import ADRDataset, get_best_device
except ImportError:
    print("Error: models.py or train.py not found in the same directory")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBED_COLS   = ["drug_encoded", "route_encoded", "dose_unit_rx_encoded"]

DATASET_DIRS = {
    "base":             PROJECT_ROOT / "processed_data",
    "min1000":          PROJECT_ROOT / "processed_data_min1000_private",
    "expanded":         PROJECT_ROOT / "processed_data_expanded",
    "min1000_expanded": PROJECT_ROOT / "processed_data_min1000_expanded",
}


class EmbeddingADRDataset(Dataset):
    """Splits feature columns into embedding indices and numerical features."""

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        self.drug_idx      = torch.LongTensor(X["drug_encoded"].values)
        self.route_idx     = torch.LongTensor(X["route_encoded"].values)
        self.dose_unit_idx = torch.LongTensor(X["dose_unit_rx_encoded"].values)
        num_cols = [c for c in X.columns if c not in EMBED_COLS]
        self.numerical = torch.FloatTensor(X[num_cols].values)
        y_arr = y.values if hasattr(y, "values") else y
        self.y = torch.FloatTensor(y_arr.flatten()).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.drug_idx[idx],
            self.route_idx[idx],
            self.dose_unit_idx[idx],
            self.numerical[idx],
            self.y[idx],
        )


def calculate_metrics_at_thresholds(y_true, y_proba, thresholds=(0.3, 0.5, 0.7)):
    results = {}
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        results[threshold] = {
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "precision":   tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "f1":          float(f1_score(y_true, y_pred)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        }
    return results


def run_inference_standard(model, loader, device, desc="Predicting"):
    """Run inference for MLP / ResNet / Attention. Returns (logits, y_true)."""
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc=desc):
            out = model(X_batch.to(device))
            all_logits.extend(out.cpu().numpy())
            all_targets.extend(y_batch.numpy())
    return np.array(all_logits).flatten(), np.array(all_targets).flatten()


def run_inference_embedding(model, loader, device, desc="Predicting"):
    """Run inference for EmbeddingMLP (four-input forward pass). Returns (logits, y_true)."""
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for d_idx, r_idx, du_idx, numerical, y_batch in tqdm(loader, desc=desc):
            d_idx, r_idx, du_idx = d_idx.to(device), r_idx.to(device), du_idx.to(device)
            out = model(d_idx, r_idx, du_idx, numerical.to(device))
            all_logits.extend(out.cpu().numpy())
            all_targets.extend(y_batch.numpy())
    return np.array(all_logits).flatten(), np.array(all_targets).flatten()


def find_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Find scalar temperature T that minimises NLL on a held-out set.

    Dividing logits by T before sigmoid corrects systematic over/under-confidence.
    T > 1 softens predictions; T < 1 sharpens them.
    """
    from sklearn.metrics import log_loss

    def nll(T):
        proba = 1.0 / (1.0 + np.exp(-logits / T))
        return log_loss(y_true, proba)

    result = minimize_scalar(nll, bounds=(0.01, 20.0), method="bounded")
    return float(result.x)


def plot_roc_curve(fpr, tpr, auroc, model_name, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auroc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}", fontweight="bold")
    plt.legend(); plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ROC curve saved to {save_path}")
    plt.close()


def plot_pr_curve(precision, recall, auprc, model_name, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"{model_name} (AUPRC={auprc:.3f})", linewidth=2)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {model_name}", fontweight="bold")
    plt.legend(); plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"PR curve saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No ADR", "ADR"], yticklabels=["No ADR", "ADR"])
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix — {model_name}", fontweight="bold")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_training_history(history, model_name, save_path=None):
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, label in zip(axes,
                                  ["loss", "auroc", "auprc"],
                                  ["Loss", "AUROC", "AUPRC"]):
        ax.plot(history[f"train_{metric}"], label=f"Train {label}", linewidth=2)
        ax.plot(history[f"val_{metric}"],   label=f"Val {label}",   linewidth=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel(label)
        ax.set_title(f"Training {label}", fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle(f"{model_name} Training History", fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history saved to {save_path}")
    plt.close()


def evaluate_single_model(model_name, data_dir, models_dir, results_dir, device,
                           exclude_cols=None, calibrate=False):
    """Load, run inference, compute metrics, and save results for one model.

    If calibrate=True, fits a temperature scalar T on the validation set and
    reports calibrated metrics alongside uncalibrated ones.
    """
    exclude_cols = exclude_cols or []
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Evaluating {model_name.upper()}")
    print(f"{'='*80}")

    # --- Load test data ---
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv")
    if exclude_cols:
        X_test = X_test.drop(columns=[c for c in exclude_cols if c in X_test.columns])
    print(f"Test set: {X_test.shape}  ADR rate: {y_test.values.mean():.3f}")

    is_embedding = (model_name == "embedding")

    if is_embedding:
        # --- EmbeddingMLP ---
        encoders_path = data_dir / "label_encoders.pkl"
        if not encoders_path.exists():
            raise FileNotFoundError(f"Missing label_encoders.pkl in {data_dir}")
        with open(encoders_path, "rb") as f:
            encoders = pickle.load(f)

        num_drugs      = len(encoders["drug"].classes_)
        num_routes     = len(encoders["route"].classes_)
        num_dose_units = len(encoders["dose_unit_rx"].classes_)
        num_numerical  = len([c for c in X_test.columns if c not in EMBED_COLS])

        model = EmbeddingMLP(
            num_drugs=num_drugs, num_routes=num_routes, num_dose_units=num_dose_units,
            embedding_dim=128, numerical_features=num_numerical,
            hidden_dims=[256, 128, 64], dropout_rate=0.3,
        ).to(device)

        model_path = models_dir / "embedding_mlp_best.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        test_loader = DataLoader(
            EmbeddingADRDataset(X_test, y_test),
            batch_size=1024, shuffle=False, num_workers=0,
        )
        logits, y_true = run_inference_embedding(model, test_loader, device)

        if calibrate:
            X_val = pd.read_csv(data_dir / "X_val.csv")
            y_val = pd.read_csv(data_dir / "y_val.csv")
            if exclude_cols:
                X_val = X_val.drop(columns=[c for c in exclude_cols if c in X_val.columns])
            val_loader = DataLoader(
                EmbeddingADRDataset(X_val, y_val),
                batch_size=1024, shuffle=False, num_workers=0,
            )
            val_logits, val_targets = run_inference_embedding(
                model, val_loader, device, desc="Val (calibration)"
            )

    else:
        # --- MLP / ResNet / Attention ---
        model_configs = {
            "mlp":       dict(input_dim=X_test.shape[1], hidden_dims=[256, 128, 64, 32], dropout_rate=0.3),
            "resnet":    dict(input_dim=X_test.shape[1], hidden_dim=256, num_blocks=3,   dropout_rate=0.3),
            "attention": dict(input_dim=X_test.shape[1], hidden_dims=[256, 128],         dropout_rate=0.3),
        }
        if model_name not in model_configs:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: mlp, resnet, attention, embedding")

        model = get_model(model_name, **model_configs[model_name]).to(device)

        model_path = models_dir / f"{model_name}_best.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        test_loader = DataLoader(
            ADRDataset(X_test, y_test),
            batch_size=1024, shuffle=False, num_workers=0,
        )
        logits, y_true = run_inference_standard(model, test_loader, device)

        if calibrate:
            X_val = pd.read_csv(data_dir / "X_val.csv")
            y_val = pd.read_csv(data_dir / "y_val.csv")
            if exclude_cols:
                X_val = X_val.drop(columns=[c for c in exclude_cols if c in X_val.columns])
            val_loader = DataLoader(
                ADRDataset(X_val, y_val),
                batch_size=1024, shuffle=False, num_workers=0,
            )
            val_logits, val_targets = run_inference_standard(
                model, val_loader, device, desc="Val (calibration)"
            )

    # --- Temperature scaling ---
    temperature = 1.0
    if calibrate:
        temperature = find_temperature(val_logits, val_targets)
        print(f"\nTemperature scaling: T = {temperature:.4f}")

    y_proba      = 1.0 / (1.0 + np.exp(-logits / temperature))
    y_proba_raw  = 1.0 / (1.0 + np.exp(-logits))  # T=1, for comparison

    # --- Metrics ---
    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    fpr, tpr, _          = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    threshold_metrics    = calculate_metrics_at_thresholds(y_true, y_proba)
    y_pred               = (y_proba >= 0.5).astype(int)

    # --- Print ---
    print(f"\nAUROC: {auroc:.4f}  AUPRC: {auprc:.4f}")
    print(f"\n{'Threshold':<12} {'Sensitivity':<14} {'Specificity':<14} {'Precision':<12} F1")
    for t, m in threshold_metrics.items():
        print(f"{t:<12.1f} {m['sensitivity']:<14.4f} {m['specificity']:<14.4f} "
              f"{m['precision']:<12.4f} {m['f1']:.4f}")
    print(f"\nClassification Report (threshold=0.5):")
    print(classification_report(y_true, y_pred, target_names=["No ADR", "ADR"], digits=4))

    if calibrate:
        print(f"\n--- Uncalibrated (T=1.0) vs Calibrated (T={temperature:.4f}) at threshold=0.5 ---")
        raw_metrics = calculate_metrics_at_thresholds(y_true, y_proba_raw, thresholds=(0.5,))
        cal_metrics = calculate_metrics_at_thresholds(y_true, y_proba,     thresholds=(0.5,))
        r, c = raw_metrics[0.5], cal_metrics[0.5]
        print(f"{'':20} {'Uncalibrated':>14} {'Calibrated':>12}")
        for key in ("sensitivity", "specificity", "precision", "f1"):
            print(f"  {key:<18} {r[key]:>14.4f} {c[key]:>12.4f}")

    # --- Plots ---
    plot_roc_curve(fpr, tpr, auroc, model_name,
                   save_path=results_dir / f"{model_name}_roc_curve.png")
    plot_pr_curve(precision, recall, auprc, model_name,
                  save_path=results_dir / f"{model_name}_pr_curve.png")
    plot_confusion_matrix(y_true, y_pred, model_name,
                          save_path=results_dir / f"{model_name}_confusion_matrix.png")

    history_path = models_dir / f"{model_name}_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        plot_training_history(history, model_name,
                              save_path=results_dir / f"{model_name}_training_history.png")

    # --- Save JSON ---
    output = {
        "model":       model_name,
        "dataset":     str(data_dir),
        "auroc":       float(auroc),
        "auprc":       float(auprc),
        "temperature": float(temperature),
        "threshold_metrics": {
            str(k): {kk: float(vv) for kk, vv in v.items() if kk not in ("tp","tn","fp","fn")}
            for k, v in threshold_metrics.items()
        },
    }
    with open(results_dir / f"{model_name}_evaluation.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_dir}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Evaluate ADR prediction models")
    parser.add_argument("--model", default="all",
                        choices=["mlp", "resnet", "attention", "embedding", "all"],
                        help="Model to evaluate (default: all = mlp+resnet+attention)")
    parser.add_argument("--dataset", default="base",
                        choices=list(DATASET_DIRS),
                        help="Processed dataset to evaluate on (default: base)")
    parser.add_argument("--models-dir", default=None,
                        help="Directory containing saved model .pth files (default: models/)")
    parser.add_argument("--results-dir", default=None,
                        help="Directory to save evaluation outputs (default: results/)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        metavar="COL",
                        help="Feature columns to exclude before evaluation")
    parser.add_argument("--calibrate", action="store_true",
                        help="Apply temperature scaling (fit on val set, report calibrated metrics)")
    args = parser.parse_args()

    data_dir    = DATASET_DIRS[args.dataset]
    models_dir  = Path(args.models_dir) if args.models_dir else PROJECT_ROOT / "models"
    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "results"
    device      = get_best_device()

    print("=" * 80)
    print("ADR Prediction Model Evaluation")
    print("=" * 80)
    print(f"Dataset:    {data_dir}")
    print(f"Models dir: {models_dir}")
    print(f"Device:     {device}")

    models_to_run = (
        ["mlp", "resnet", "attention"] if args.model == "all" else [args.model]
    )

    all_results = {}
    for model_name in models_to_run:
        try:
            result = evaluate_single_model(
                model_name, data_dir, models_dir, results_dir, device,
                exclude_cols=args.exclude, calibrate=args.calibrate,
            )
            all_results[model_name] = result
        except FileNotFoundError as e:
            print(f"Skipping {model_name}: {e}")

    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Model':<12} {'AUROC':<10} {'AUPRC':<10} {'Sens@0.5':<12} {'Spec@0.5':<12} F1@0.5")
        for name, r in all_results.items():
            t = r["threshold_metrics"]["0.5"]
            print(f"{name:<12} {r['auroc']:<10.4f} {r['auprc']:<10.4f} "
                  f"{t['sensitivity']:<12.4f} {t['specificity']:<12.4f} {t['f1']:.4f}")

    print("\n" + "=" * 80)
    print("Evaluation complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
