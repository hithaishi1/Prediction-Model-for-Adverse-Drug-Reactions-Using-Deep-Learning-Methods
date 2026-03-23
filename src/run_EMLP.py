"""
Training script for the EmbeddingMLP model.

Uses embedding layers for drug, route, and dose unit, and a standard MLP
for the remaining numerical features. Designed for the expanded feature
datasets produced by preprocessing.py.

Usage:
    python src/embeddingMLP.py
    python src/embeddingMLP.py --dataset min1000_expanded
    python src/embeddingMLP.py --dataset expanded --exclude prior_adr icu_stay
    python src/embeddingMLP.py --embedding-dim 64 --hidden-dims 512 256 128
"""

import sys
import os
import argparse
import json
import pickle
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from models import EmbeddingMLP, FocalLoss
from train import get_best_device, set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42

DATASET_DIRS = {
    "min1000_expanded": PROJECT_ROOT / "processed_data_min1000_expanded",
    "expanded":         PROJECT_ROOT / "processed_data_expanded",
    "min1000":          PROJECT_ROOT / "processed_data_min1000",
    "base":             PROJECT_ROOT / "processed_data",
}

EMBED_COLS = ["drug_encoded", "route_encoded", "dose_unit_rx_encoded"]


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


def main():
    parser = argparse.ArgumentParser(description="Train EmbeddingMLP for ADR prediction")
    parser.add_argument("--dataset", default="min1000_expanded",
                        choices=list(DATASET_DIRS),
                        help="Processed dataset to train on (default: min1000_expanded)")
    parser.add_argument("--exclude", nargs="*", default=["prior_adr", "icu_stay"],
                        metavar="COL",
                        help="Feature columns to exclude (default: prior_adr icu_stay)")
    parser.add_argument("--embedding-dim", type=int, default=128,
                        help="Drug embedding dimension (default: 128)")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128, 64],
                        metavar="N",
                        help="Hidden layer sizes (default: 256 128 64)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate (default: 0.3)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size (default: 1024)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum epochs (default: 50)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (default: 10)")
    args = parser.parse_args()

    processed_dir = DATASET_DIRS[args.dataset]
    exclude_cols  = args.exclude or []

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EmbeddingMLP Training")
    print("=" * 80)
    print(f"Dataset:  {processed_dir}")
    print(f"Excluded: {exclude_cols or 'none'}")
    print(f"Output:   {models_dir}")

    # Validate inputs exist
    for f in ["X_train.csv", "y_train.csv", "X_val.csv", "y_val.csv", "label_encoders.pkl"]:
        if not (processed_dir / f).exists():
            raise FileNotFoundError(f"Missing {f} in {processed_dir}. Run preprocessing.py first.")

    # Load data
    print("\n[1/4] Loading data...")
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv")
    X_val   = pd.read_csv(processed_dir / "X_val.csv")
    y_val   = pd.read_csv(processed_dir / "y_val.csv")

    if exclude_cols:
        missing = [c for c in exclude_cols if c not in X_train.columns]
        if missing:
            raise ValueError(f"Columns not found in dataset: {missing}")
        X_train = X_train.drop(columns=exclude_cols)
        X_val   = X_val.drop(columns=exclude_cols)
        print(f"Excluded columns: {exclude_cols}")

    with open(processed_dir / "label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    num_drugs      = len(encoders["drug"].classes_)
    num_routes     = len(encoders["route"].classes_)
    num_dose_units = len(encoders["dose_unit_rx"].classes_)
    num_numerical  = len([c for c in X_train.columns if c not in EMBED_COLS])

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  "
          f"ADR rate: {y_train.values.mean():.3f}")
    print(f"Vocab — drugs: {num_drugs}, routes: {num_routes}, "
          f"dose units: {num_dose_units}, numerical: {num_numerical}")

    # Build data loaders
    print("\n[2/4] Building data loaders...")
    device = get_best_device()
    set_global_seed(RANDOM_STATE)

    loader_gen = torch.Generator().manual_seed(RANDOM_STATE)
    train_loader = DataLoader(
        EmbeddingADRDataset(X_train, y_train),
        batch_size=args.batch_size, shuffle=True, num_workers=0, generator=loader_gen,
    )
    val_loader = DataLoader(
        EmbeddingADRDataset(X_val, y_val),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    # Build model
    print("\n[3/4] Building model...")
    model = EmbeddingMLP(
        num_drugs=num_drugs,
        num_routes=num_routes,
        num_dose_units=num_dose_units,
        embedding_dim=args.embedding_dim,
        numerical_features=num_numerical,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout,
    ).to(device)

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

    print(f"Device:     {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Hidden dims: {args.hidden_dims}  Embedding dim: {args.embedding_dim}  "
          f"Dropout: {args.dropout}  LR: {args.lr}")

    # Train
    print("\n[4/4] Training...")
    print("=" * 80)

    history = {
        "train_loss": [], "val_loss": [],
        "train_auroc": [], "val_auroc": [],
        "train_auprc": [], "val_auprc": [],
    }
    best_auroc       = 0.0
    patience_counter = 0
    best_state       = None

    for epoch in range(args.epochs):
        # --- train ---
        model.train()
        total_loss, all_preds, all_targets = 0.0, [], []
        for d_idx, r_idx, du_idx, numerical, y_batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} train"
        ):
            d_idx, r_idx, du_idx = d_idx.to(device), r_idx.to(device), du_idx.to(device)
            numerical, y_batch = numerical.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(d_idx, r_idx, du_idx, numerical)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(out).detach().cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

        train_auroc = roc_auc_score(all_targets, all_preds)
        train_auprc = average_precision_score(all_targets, all_preds)

        # --- validate ---
        model.eval()
        vtotal_loss, vpreds, vtargets = 0.0, [], []
        with torch.no_grad():
            for d_idx, r_idx, du_idx, numerical, y_batch in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} val"
            ):
                d_idx, r_idx, du_idx = d_idx.to(device), r_idx.to(device), du_idx.to(device)
                numerical, y_batch = numerical.to(device), y_batch.to(device)
                out = model(d_idx, r_idx, du_idx, numerical)
                vtotal_loss += criterion(out, y_batch).item()
                vpreds.extend(torch.sigmoid(out).cpu().numpy())
                vtargets.extend(y_batch.cpu().numpy())

        val_auroc = roc_auc_score(vtargets, vpreds)
        val_auprc = average_precision_score(vtargets, vpreds)

        scheduler.step(val_auroc)

        history["train_loss"].append(total_loss / len(train_loader))
        history["val_loss"].append(vtotal_loss / len(val_loader))
        history["train_auroc"].append(train_auroc)
        history["val_auroc"].append(val_auroc)
        history["train_auprc"].append(train_auprc)
        history["val_auprc"].append(val_auprc)

        print(f"Epoch {epoch + 1:>3}: "
              f"train_auroc={train_auroc:.4f}  val_auroc={val_auroc:.4f}  "
              f"val_auprc={val_auprc:.4f}")

        if val_auroc > best_auroc + 0.001:
            best_auroc       = val_auroc
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1} "
                      f"(no improvement for {args.patience} epochs)")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Save
    model_path   = models_dir / "embedding_mlp_best.pth"
    history_path = models_dir / "embedding_mlp_history.json"
    torch.save({"model_state_dict": model.state_dict()}, model_path)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "dataset":        args.dataset,
        "excluded_cols":  exclude_cols,
        "device":         device,
        "train_rows":     int(len(X_train)),
        "val_rows":       int(len(X_val)),
        "num_drugs":      num_drugs,
        "num_routes":     num_routes,
        "num_dose_units": num_dose_units,
        "num_numerical":  num_numerical,
        "embedding_dim":  args.embedding_dim,
        "hidden_dims":    args.hidden_dims,
        "dropout":        args.dropout,
        "lr":             args.lr,
        "best_val_auroc": float(best_auroc),
        "best_val_auprc": float(max(history["val_auprc"])),
        "epochs_trained": len(history["val_auroc"]),
        "model_path":     str(model_path),
        "history_path":   str(history_path),
    }
    with open(models_dir / "embedding_mlp_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Best val AUROC: {best_auroc:.4f}  |  Best val AUPRC: {max(history['val_auprc']):.4f}")
    print(f"Results saved to: {models_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
