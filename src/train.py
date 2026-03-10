"""
Training script for ADR prediction models
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
# Allows running this file directly (`python src/train.py`) without package install.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tqdm import tqdm
import pickle
import json
from datetime import datetime
import random

# Import models
try:
    from models import get_model, FocalLoss
except ImportError:
    print("Error: models.py not found in the same directory as train.py")
    print("Please ensure models.py is in the src/ directory")
    sys.exit(1)


def set_global_seed(seed):
    """Set random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ADRDataset(Dataset):
    """PyTorch Dataset for ADR prediction"""
    def __init__(self, X, y):
        # Accept either pandas objects or numpy arrays and normalize to tensors.
        self.X = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        # Ensure y is 1D, then add dimension for BCE loss
        y_array = y.values if hasattr(y, 'values') else y
        self.y = torch.FloatTensor(y_array.flatten()).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score, model):
        # Snapshot best model state whenever monitored metric improves.
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
        elif self._is_improvement(score):
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta
    
    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


class Trainer:
    """Training manager for deep learning models"""
    
    def __init__(self, 
                 model, 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate=0.001,
                 loss_type='focal',
                 class_weights=None):
        
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        # Focal loss is default because ADR labels are often imbalanced.
        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        elif loss_type == 'bce':
            if class_weights is not None:
                pos_weight = torch.FloatTensor([class_weights[1] / class_weights[0]]).to(device)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Optimizer
        # Adam is a stable default for tabular neural nets with mixed feature scales.
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        # Reduce LR when validation AUROC plateaus to stabilize late training.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_auroc': [], 'val_auroc': [],
            'train_auprc': [], 'val_auprc': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc='Training')
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            # Keep probability-space predictions so AUROC/AUPRC are threshold-free.
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        auroc = roc_auc_score(all_targets, all_preds)
        auprc = average_precision_score(all_targets, all_preds)
        
        return avg_loss, auroc, auprc
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc='Validation'):
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        auroc = roc_auc_score(all_targets, all_preds)
        auprc = average_precision_score(all_targets, all_preds)
        
        return avg_loss, auroc, auprc
    
    def fit(self, train_loader, val_loader, epochs=50, early_stopping_patience=10):
        """Train model with early stopping"""
        
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
        
        print(f"\nTraining on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*80)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_auroc, train_auprc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_auroc, val_auprc = self.validate(val_loader)
            
            # Update learning rate
            # Scheduler monitors validation AUROC, not loss, to align with objective.
            self.scheduler.step(val_auroc)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auroc'].append(train_auroc)
            self.history['val_auroc'].append(val_auroc)
            self.history['train_auprc'].append(train_auprc)
            self.history['val_auprc'].append(val_auprc)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train AUROC: {train_auroc:.4f} | Val AUROC: {val_auroc:.4f}")
            print(f"Train AUPRC: {train_auprc:.4f} | Val AUPRC: {val_auprc:.4f}")
            
            # Early stopping
            if early_stopping(val_auroc, self.model):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        # Restore highest-val-AUROC checkpoint before returning.
        early_stopping.load_best_model(self.model)
        print(f"\nBest validation AUROC: {early_stopping.best_score:.4f}")
        
        return self.history

    def set_learning_rate(self, learning_rate):
        """Update optimizer learning rate and reset scheduler for a new training phase"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

        # Reset scheduler state so fine-tuning has its own plateau tracking window.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
    
    def predict(self, test_loader):
        """Generate predictions"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in tqdm(test_loader, desc='Predicting'):
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_targets.extend(y_batch.numpy())
        
        return np.array(all_preds), np.array(all_targets)
    
    def save_model(self, filepath):
        """Save model and training history"""
        # Save both optimizer + history so training can be resumed/analyzed.
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model and training history"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Model loaded from {filepath}")


def main():
    """Main training pipeline"""
    
    # Configuration
    # Paths are repo-relative for portability between local environments.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "processed_data"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Hyperparameters
    # These are conservative defaults; tune by validation AUROC/AUPRC.
    BATCH_SIZE = 1024
    LEARNING_RATE = 0.001
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    ENABLE_FINE_TUNING = True
    FINE_TUNE_LR_FACTOR = 0.1
    FINE_TUNE_EPOCHS = 15
    FINE_TUNE_PATIENCE = 5
    RANDOM_SEED = 42
    
    print("="*80)
    print("ADR Prediction Model Training")
    print("="*80)

    set_global_seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")
    
    # Load data
    print("\n[1/6] Loading preprocessed data...")
    required_inputs = [
        DATA_DIR / "X_train.csv",
        DATA_DIR / "y_train.csv",
        DATA_DIR / "X_val.csv",
        DATA_DIR / "y_val.csv",
    ]
    missing_inputs = [str(path) for path in required_inputs if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Missing required input files:\n- " + "\n- ".join(missing_inputs)
        )

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")
    X_val = pd.read_csv(DATA_DIR / "X_val.csv")
    y_val = pd.read_csv(DATA_DIR / "y_val.csv")
    
    print(f"Train set: {X_train.shape}")
    print(f"Val set: {X_val.shape}")
    print(f"ADR rate - Train: {y_train.mean().values[0]:.3f}, Val: {y_val.mean().values[0]:.3f}")
    
    # Create datasets
    print("\n[2/6] Creating PyTorch datasets...")
    train_dataset = ADRDataset(X_train, y_train)
    val_dataset = ADRDataset(X_val, y_val)

    loader_generator = torch.Generator().manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=loader_generator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Calculate class weights
    # Computed for reference and optional BCE weighting if needed.
    y_train_flat = y_train.values.flatten() if hasattr(y_train, 'values') else y_train.flatten()
    unique, counts = np.unique(y_train_flat, return_counts=True)
    class_counts = dict(zip(unique, counts))
    class_weights = {0: 1.0, 1: class_counts[0] / class_counts[1]}
    print(f"Class weights: {class_weights}")
    
    # Train different models
    # All models consume the same tabular feature matrix for fair comparison.
    models_config = {
        'mlp': {
            'model_type': 'mlp',
            'input_dim': X_train.shape[1],
            'hidden_dims': [256, 128, 64, 32],
            'dropout_rate': 0.3
        },
        'resnet': {
            'model_type': 'resnet',
            'input_dim': X_train.shape[1],
            'hidden_dim': 256,
            'num_blocks': 3,
            'dropout_rate': 0.3
        },
        'attention': {
            'model_type': 'attention',
            'input_dim': X_train.shape[1],
            'hidden_dims': [256, 128],
            'dropout_rate': 0.3
        }
    }
    
    results = {}
    
    for model_name, config in models_config.items():
        print(f"\n{'='*80}")
        print(f"[3/6] Training {model_name.upper()} model...")
        print(f"{'='*80}")
        
        # Create model
        # Factory function keeps architecture selection declarative.
        model_type = config.pop('model_type')
        model = get_model(model_type, **config)
        
        # Create trainer
        # Using focal loss here; class_weights currently not passed.
        trainer = Trainer(
            model, 
            learning_rate=LEARNING_RATE,
            loss_type='focal',
            class_weights=None  # Using focal loss instead
        )
        
        # Train
        history = trainer.fit(
            train_loader, 
            val_loader, 
            epochs=EPOCHS,
            early_stopping_patience=EARLY_STOPPING_PATIENCE
        )

        # Optional second-phase fine-tuning at lower learning rate.
        if ENABLE_FINE_TUNING:
            fine_tune_lr = LEARNING_RATE * FINE_TUNE_LR_FACTOR
            print(f"\n[4/6] Fine-tuning {model_name.upper()} at lr={fine_tune_lr}...")
            trainer.set_learning_rate(fine_tune_lr)
            history = trainer.fit(
                train_loader,
                val_loader,
                epochs=FINE_TUNE_EPOCHS,
                early_stopping_patience=FINE_TUNE_PATIENCE
            )
        
        # Save model
        model_path = MODELS_DIR / f"{model_name}_best.pth"
        trainer.save_model(model_path)
        
        # Store results
        results[model_name] = {
            'best_val_auroc': max(history['val_auroc']),
            'best_val_auprc': max(history['val_auprc']),
            'history': history
        }
        
        # Save history
        # Per-model history file drives later plotting in evaluation script.
        with open(MODELS_DIR / f"{model_name}_history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    # Save results summary
    print("\n" + "="*80)
    print("[5/6] Training Summary")
    print("="*80)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Best Val AUROC: {metrics['best_val_auroc']:.4f}")
        print(f"  Best Val AUPRC: {metrics['best_val_auprc']:.4f}")
    
    with open(MODELS_DIR / "training_results.json", 'w') as f:
        results_serializable = {
            k: {
                'best_val_auroc': v['best_val_auroc'],
                'best_val_auprc': v['best_val_auprc']
            } 
            for k, v in results.items()
        }
        json.dump(results_serializable, f, indent=2)
    
    print("\n" + "="*80)
    print("[6/6] Training complete!")
    print(f"Models saved to: {MODELS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
