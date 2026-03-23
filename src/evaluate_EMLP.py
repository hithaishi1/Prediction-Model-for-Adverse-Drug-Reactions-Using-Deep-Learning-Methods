"""
Evaluation script for the EmbeddingMLP ADR prediction model.

Loads the trained EmbeddingMLP from models/embedding_mlp_best.pth and
evaluates it on the min1000 expanded test set. Saves metrics, plots,
and a JSON results file to results/.

Usage:
    python src/evaluate_EMLP.py
"""

import sys
import os
from pathlib import Path

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
    classification_report
)
from torch.utils.data import Dataset, DataLoader
import json
import pickle
from tqdm import tqdm

try:
    from models import EmbeddingMLP
    from train import get_best_device
except ImportError:
    print("Error: models.py or train.py not found in the same directory")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBED_COLS   = ["drug_encoded", "route_encoded", "dose_unit_rx_encoded"]
DEFAULT_EXCLUDE = ["prior_adr", "icu_stay"]


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


def plot_roc_curve(fpr, tpr, auroc, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'EmbeddingMLP (AUC = {auroc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - EmbeddingMLP ADR Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    plt.close()


def plot_pr_curve(precision, recall, auprc, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'EmbeddingMLP (AUPRC = {auprc:.3f})', linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve - EmbeddingMLP ADR Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No ADR', 'ADR'],
                yticklabels=['No ADR', 'ADR'])
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix - EmbeddingMLP', fontsize=14, fontweight='bold')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.close()


def calculate_metrics_at_thresholds(y_true, y_proba, thresholds=[0.3, 0.5, 0.7]):
    results = {}
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        results[threshold] = {
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'precision':   tp / (tp + fp) if (tp + fp) > 0 else 0,
            'f1':          f1_score(y_true, y_pred),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        }
    return results


def main():
    DATA_DIR   = PROJECT_ROOT / "processed_data_min1000_expanded"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "embedding_mlp_best.pth"

    print("=" * 80)
    print("EmbeddingMLP Evaluation")
    print("=" * 80)

    for path in [DATA_DIR / "X_test.csv", DATA_DIR / "y_test.csv", model_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}\n"
                "Run preprocessing.py then run_EMLP.py first."
            )

    device = get_best_device()
    print(f"Device: {device}")

    # Load test data
    print("\n[1/4] Loading test data...")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")

    X_test = X_test.drop(columns=[c for c in DEFAULT_EXCLUDE if c in X_test.columns])
    print(f"Test set: {X_test.shape}  ADR rate: {y_test.values.mean():.3f}")

    with open(DATA_DIR / "label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    num_drugs      = len(encoders["drug"].classes_)
    num_routes     = len(encoders["route"].classes_)
    num_dose_units = len(encoders["dose_unit_rx"].classes_)
    num_numerical  = len([c for c in X_test.columns if c not in EMBED_COLS])

    # Load model
    print("\n[2/4] Loading model...")
    model = EmbeddingMLP(
        num_drugs=num_drugs,
        num_routes=num_routes,
        num_dose_units=num_dose_units,
        embedding_dim=128,
        numerical_features=num_numerical,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.3,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded: {model_path}")

    # Run inference
    print("\n[3/4] Running inference...")
    test_loader = DataLoader(
        EmbeddingADRDataset(X_test, y_test),
        batch_size=1024, shuffle=False, num_workers=4,
    )

    all_preds, all_targets = [], []
    with torch.no_grad():
        for d_idx, r_idx, du_idx, numerical, y_batch in tqdm(test_loader, desc="Predicting"):
            d_idx, r_idx, du_idx = d_idx.to(device), r_idx.to(device), du_idx.to(device)
            out = model(d_idx, r_idx, du_idx, numerical.to(device))
            all_preds.extend(torch.sigmoid(out).cpu().numpy())
            all_targets.extend(y_batch.numpy())

    y_true  = np.array(all_targets).flatten()
    y_proba = np.array(all_preds).flatten()
    y_pred  = (y_proba >= 0.5).astype(int)

    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    fpr, tpr, _       = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    threshold_metrics = calculate_metrics_at_thresholds(y_true, y_proba)

    # Print report
    print("\n[4/4] Results")
    print("=" * 80)
    print(f"{'AUROC':<10} {'AUPRC':<10} {'F1@0.5':<10} {'Sens@0.5':<12} {'Spec@0.5'}")
    t = threshold_metrics[0.5]
    print(f"{auroc:<10.4f} {auprc:<10.4f} {t['f1']:<10.4f} "
          f"{t['sensitivity']:<12.4f} {t['specificity']:.4f}")

    print("\nPerformance at Different Thresholds:")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<10} {'Specificity'}")
    for thresh, m in threshold_metrics.items():
        print(f"{thresh:<12.1f} {m['precision']:<12.4f} {m['sensitivity']:<12.4f} "
              f"{m['f1']:<10.4f} {m['specificity']:.4f}")

    print("\nClassification Report (Threshold = 0.5):")
    print(classification_report(y_true, y_pred, target_names=['No ADR', 'ADR'], digits=4))

    # Save plots
    plot_roc_curve(fpr, tpr, auroc,
                   save_path=RESULTS_DIR / "embedding_mlp_roc_curve.png")
    plot_pr_curve(precision, recall, auprc,
                  save_path=RESULTS_DIR / "embedding_mlp_pr_curve.png")
    plot_confusion_matrix(y_true, y_pred,
                          save_path=RESULTS_DIR / "embedding_mlp_confusion_matrix.png")

    # Save JSON
    results = {
        "model": "EmbeddingMLP",
        "dataset": "min1000_expanded",
        "excluded_cols": DEFAULT_EXCLUDE,
        "test_rows": int(len(y_true)),
        "adr_rate": float(y_true.mean()),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "threshold_metrics": {
            str(k): {
                "sensitivity": float(v["sensitivity"]),
                "specificity": float(v["specificity"]),
                "precision":   float(v["precision"]),
                "f1":          float(v["f1"]),
            }
            for k, v in threshold_metrics.items()
        },
    }
    out_path = RESULTS_DIR / "embedding_mlp_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
