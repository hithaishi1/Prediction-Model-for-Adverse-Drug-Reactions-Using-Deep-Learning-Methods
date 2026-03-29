"""
Hyperparameter tuning for ADR prediction models using Optuna.

Tunes the MLP classifier on the min1000 expanded feature dataset.
Searches over learning rate, dropout, focal loss alpha/gamma, and
hidden layer width. Results are saved to results/tuning/.

Progress is saved to an SQLite database so runs can be stopped and
resumed. Each invocation adds --trials more trials to the same study.

Usage:
    python src/tune.py                    # run 20 trials (resumable)
    python src/tune.py --trials 50        # run 50 trials
    python src/tune.py --dataset expanded # full expanded dataset
    python src/tune.py --reset            # delete saved study and start fresh
"""

import sys
import os
import argparse
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import optuna
from optuna.samplers import TPESampler

from models import MLPClassifier, FocalLoss
from train import ADRDataset, get_best_device, set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_STATE = 42
EXCLUDE_COLS = ["prior_adr", "icu_stay"]


def load_dataset(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val splits for the given dataset variant."""
    dirs = {
        "min1000_expanded": PROJECT_ROOT / "processed_data_min1000_expanded",
        "expanded":         PROJECT_ROOT / "processed_data_expanded",
        "min1000":          PROJECT_ROOT / "processed_data_min1000",
        "base":             PROJECT_ROOT / "processed_data",
    }
    if dataset not in dirs:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(dirs)}")

    d = dirs[dataset]
    for f in ["X_train.csv", "y_train.csv", "X_val.csv", "y_val.csv"]:
        if not (d / f).exists():
            raise FileNotFoundError(f"Missing {f} in {d}. Run preprocessing.py first.")

    X_train = pd.read_csv(d / "X_train.csv")
    y_train = pd.read_csv(d / "y_train.csv")
    X_val   = pd.read_csv(d / "X_val.csv")
    y_val   = pd.read_csv(d / "y_val.csv")

    # Drop leaky/excluded columns
    X_train = X_train.drop(columns=[c for c in EXCLUDE_COLS if c in X_train.columns])
    X_val   = X_val.drop(columns=[c for c in EXCLUDE_COLS if c in X_val.columns])
    return X_train, y_train, X_val, y_val


def run_trial(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    device: str,
) -> float:
    """Train one configuration and return validation AUROC."""

    # --- Hyperparameter search space ---
    lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout      = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    focal_alpha  = trial.suggest_float("focal_alpha", 0.25, 0.75, step=0.25)
    focal_gamma  = trial.suggest_float("focal_gamma", 1.0, 3.0, step=1.0)
    width        = trial.suggest_categorical("width", [128, 256, 512])
    depth        = trial.suggest_int("depth", 2, 4)
    batch_size   = trial.suggest_categorical("batch_size", [512, 1024, 2048])

    hidden_dims = [width // (2 ** i) for i in range(depth)]
    # Clamp minimum layer size to 32 to avoid overly narrow final layers
    hidden_dims = [max(32, d) for d in hidden_dims]

    print(f"\n{'='*60}")
    print(f"Trial {trial.number}  |  lr={lr:.5f}  dropout={dropout}  "
          f"alpha={focal_alpha}  gamma={focal_gamma}")
    print(f"             |  hidden={hidden_dims}  batch={batch_size}")
    print(f"{'='*60}")

    set_global_seed(RANDOM_STATE)
    loader_gen = torch.Generator().manual_seed(RANDOM_STATE)

    train_loader = DataLoader(
        ADRDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True, num_workers=0, generator=loader_gen,
    )
    val_loader = DataLoader(
        ADRDataset(X_val, y_val),
        batch_size=batch_size, shuffle=False, num_workers=0,
    )

    model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout_rate=dropout,
    ).to(device)

    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_auroc = 0.0
    patience_counter = 0
    PATIENCE = 7
    MAX_EPOCHS = 30  # Capped for tuning speed; full training uses 50

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validate
        model.eval()
        vpreds, vtargets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                vpreds.extend(torch.sigmoid(model(X_batch)).cpu().numpy())
                vtargets.extend(y_batch.numpy())

        val_auroc = roc_auc_score(vtargets, vpreds)
        avg_loss = epoch_loss / len(train_loader)
        marker = " *" if val_auroc > best_val_auroc + 0.001 else ""
        print(f"  Epoch {epoch+1:02d}/{MAX_EPOCHS}  loss={avg_loss:.4f}  "
              f"val_auroc={val_auroc:.4f}{marker}")

        # Optuna pruning: abort unpromising trials early
        trial.report(val_auroc, epoch)
        if trial.should_prune():
            print(f"  Pruned at epoch {epoch+1}.")
            raise optuna.exceptions.TrialPruned()

        if val_auroc > best_val_auroc + 0.001:
            best_val_auroc = val_auroc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop (patience={PATIENCE}).")
                break

    print(f"  => Best val AUROC: {best_val_auroc:.4f}")
    return best_val_auroc


def tune(n_trials: int = 20, dataset: str = "min1000_expanded", reset: bool = False) -> None:
    """Run the Optuna study and save results. Resumes automatically if a prior study exists."""
    results_dir = PROJECT_ROOT / "results" / "tuning"
    results_dir.mkdir(parents=True, exist_ok=True)

    db_path = results_dir / f"tuning_{dataset}.db"
    storage = f"sqlite:///{db_path}"
    study_name = f"mlp_{dataset}"

    if reset and db_path.exists():
        db_path.unlink()
        print(f"Deleted existing study: {db_path}")

    device = get_best_device()
    print(f"Device: {device}")
    print(f"Dataset: {dataset}")
    print(f"Trials this run: {n_trials}")

    X_train, y_train, X_val, y_val = load_dataset(dataset)
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  "
          f"ADR rate: {y_train.values.mean():.3f}")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    completed_before = len([t for t in study.trials if t.value is not None])
    if completed_before > 0:
        print(f"Resuming study — {completed_before} trials already completed.")

    study.optimize(
        lambda trial: run_trial(trial, X_train, y_train, X_val, y_val, device),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"  Val AUROC: {best.value:.4f}")
    print(f"  Params:    {best.params}")

    # Save full results
    total_completed = len([t for t in study.trials if t.value is not None])
    summary = {
        "dataset": dataset,
        "total_trials_completed": total_completed,
        "best_val_auroc": best.value,
        "best_params": best.params,
        "all_trials": [
            {
                "number": t.number,
                "val_auroc": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    out_path = results_dir / f"tuning_{dataset}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print top-5 trials
    completed = [t for t in study.trials if t.value is not None]
    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print("\nTop 5 trials:")
    print(f"  {'#':<5} {'AUROC':<10} {'lr':<10} {'dropout':<10} {'alpha':<8} {'gamma':<8} {'width':<8} {'depth':<6} {'batch'}")
    for t in top5:
        p = t.params
        print(f"  {t.number:<5} {t.value:<10.4f} "
              f"{p['lr']:<10.5f} {p['dropout']:<10.1f} "
              f"{p['focal_alpha']:<8.2f} {p['focal_gamma']:<8.1f} "
              f"{p['width']:<8} {p['depth']:<6} {p['batch_size']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for ADR MLP")
    parser.add_argument("--trials",  type=int, default=20,
                        help="Number of trials to run this session (default: 20)")
    parser.add_argument("--dataset", type=str, default="min1000_expanded",
                        choices=["min1000_expanded", "expanded", "min1000", "base"],
                        help="Which processed dataset to tune on (default: min1000_expanded)")
    parser.add_argument("--reset", action="store_true",
                        help="Delete saved study and start fresh")
    args = parser.parse_args()
    tune(n_trials=args.trials, dataset=args.dataset, reset=args.reset)
