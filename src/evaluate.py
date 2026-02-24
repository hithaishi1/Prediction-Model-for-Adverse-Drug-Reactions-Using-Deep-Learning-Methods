"""
Comprehensive evaluation script for ADR prediction models
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report
)
from torch.utils.data import DataLoader
import json
from tqdm import tqdm

# Import local modules
try:
    from models import get_model
    from train import ADRDataset, Trainer
except ImportError:
    print("Error: models.py or train.py not found in the same directory")
    print("Please ensure all files are in the src/ directory")
    sys.exit(1)


def plot_roc_curves(results_dict, save_path=None):
    """Plot ROC curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for model_name, metrics in results_dict.items():
        fpr, tpr = metrics['fpr'], metrics['tpr']
        auc = metrics['auroc']
        plt.plot(fpr, tpr, label=f'{model_name.upper()} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - ADR Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    plt.show()


def plot_pr_curves(results_dict, save_path=None):
    """Plot Precision-Recall curves for all models"""
    plt.figure(figsize=(10, 8))
    
    for model_name, metrics in results_dict.items():
        precision, recall = metrics['precision'], metrics['recall']
        auprc = metrics['auprc']
        plt.plot(recall, precision, label=f'{model_name.upper()} (AUPRC = {auprc:.3f})', linewidth=2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - ADR Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curves saved to {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No ADR', 'ADR'],
                yticklabels=['No ADR', 'ADR'])
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def plot_training_history(history, model_name, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # AUROC
    axes[1].plot(history['train_auroc'], label='Train AUROC', linewidth=2)
    axes[1].plot(history['val_auroc'], label='Val AUROC', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('AUROC', fontsize=12)
    axes[1].set_title('AUROC Score', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # AUPRC
    axes[2].plot(history['train_auprc'], label='Train AUPRC', linewidth=2)
    axes[2].plot(history['val_auprc'], label='Val AUPRC', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('AUPRC', fontsize=12)
    axes[2].set_title('AUPRC Score', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.suptitle(f'{model_name.upper()} Training History', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    plt.show()


def calculate_metrics_at_thresholds(y_true, y_proba, thresholds=[0.3, 0.5, 0.7]):
    """Calculate metrics at different thresholds"""
    results = {}
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(y_true, y_pred)
        
        results[threshold] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
        }
    
    return results


def evaluate_model(model, test_loader, device, model_name):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_targets = []
    
    print(f"\nEvaluating {model_name.upper()}...")
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc='Predicting'):
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_targets.extend(y_batch.numpy())
    
    y_true = np.array(all_targets).flatten()
    y_proba = np.array(all_preds).flatten()
    
    # Calculate metrics
    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    # Metrics at different thresholds
    threshold_metrics = calculate_metrics_at_thresholds(y_true, y_proba)
    
    # Default threshold (0.5)
    y_pred_default = (y_proba >= 0.5).astype(int)
    
    results = {
        'auroc': auroc,
        'auprc': auprc,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'y_true': y_true,
        'y_proba': y_proba,
        'y_pred_default': y_pred_default,
        'threshold_metrics': threshold_metrics
    }
    
    return results


def print_evaluation_report(results_dict):
    """Print comprehensive evaluation report"""
    print("\n" + "="*80)
    print("MODEL EVALUATION REPORT")
    print("="*80)
    
    # Summary table
    print("\nOverall Performance:")
    print("-" * 80)
    print(f"{'Model':<15} {'AUROC':<10} {'AUPRC':<10} {'F1@0.5':<10} {'Sens@0.5':<12} {'Spec@0.5':<12}")
    print("-" * 80)
    
    for model_name, metrics in results_dict.items():
        threshold_metrics = metrics['threshold_metrics'][0.5]
        print(f"{model_name.upper():<15} "
              f"{metrics['auroc']:<10.4f} "
              f"{metrics['auprc']:<10.4f} "
              f"{threshold_metrics['f1']:<10.4f} "
              f"{threshold_metrics['sensitivity']:<12.4f} "
              f"{threshold_metrics['specificity']:<12.4f}")
    
    # Detailed metrics for best model
    best_model = max(results_dict.items(), key=lambda x: x[1]['auroc'])
    model_name, metrics = best_model
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {model_name.upper()}")
    print(f"{'='*80}")
    
    print("\nPerformance at Different Thresholds:")
    print("-" * 80)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<10} {'Specificity':<12}")
    print("-" * 80)
    
    for threshold, threshold_metrics in metrics['threshold_metrics'].items():
        print(f"{threshold:<12.1f} "
              f"{threshold_metrics['precision']:<12.4f} "
              f"{threshold_metrics['sensitivity']:<12.4f} "
              f"{threshold_metrics['f1']:<10.4f} "
              f"{threshold_metrics['specificity']:<12.4f}")
    
    print("\nClassification Report (Threshold = 0.5):")
    print("-" * 80)
    print(classification_report(
        metrics['y_true'], 
        metrics['y_pred_default'],
        target_names=['No ADR', 'ADR'],
        digits=4
    ))


def main():
    """Main evaluation pipeline"""
    
    # Configuration
    BASE_DIR = "/Users/hithaishireddy/Desktop/ADR-project/Prediction-Model-for-Adverse-Drug-Reactions-Using-Deep-Learning-Methods"
    DATA_DIR = f"{BASE_DIR}/processed_data"
    MODELS_DIR = f"{BASE_DIR}/models"
    RESULTS_DIR = f"{BASE_DIR}/results"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 1024
    
    print("="*80)
    print("ADR Prediction Model Evaluation")
    print("="*80)
    
    # Load test data
    print("\n[1/4] Loading test data...")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv")
    
    print(f"Test set: {X_test.shape}")
    print(f"ADR rate: {y_test.mean().values[0]:.3f}")
    
    # Create test dataset
    test_dataset = ADRDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Evaluate all models
    print("\n[2/4] Evaluating models...")
    
    models_to_eval = {
        'mlp': {
            'model_type': 'mlp',
            'input_dim': X_test.shape[1],
            'hidden_dims': [256, 128, 64, 32],
            'dropout_rate': 0.3
        },
        'resnet': {
            'model_type': 'resnet',
            'input_dim': X_test.shape[1],
            'hidden_dim': 256,
            'num_blocks': 3,
            'dropout_rate': 0.3
        },
        'attention': {
            'model_type': 'attention',
            'input_dim': X_test.shape[1],
            'hidden_dims': [256, 128],
            'dropout_rate': 0.3
        }
    }
    
    evaluation_results = {}
    
    for model_name, config in models_to_eval.items():
        # Load model
        model_path = f"{MODELS_DIR}/{model_name}_best.pth"
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}. Skipping...")
            continue
        
        model_type = config.pop('model_type')
        model = get_model(model_type, **config)
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Evaluate
        results = evaluate_model(model, test_loader, device, model_name)
        evaluation_results[model_name] = results
        
        # Plot training history
        history_path = f"{MODELS_DIR}/{model_name}_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            plot_training_history(
                history, 
                model_name, 
                save_path=f"{RESULTS_DIR}/{model_name}_training_history.png"
            )
        
        # Plot confusion matrix
        plot_confusion_matrix(
            results['y_true'],
            results['y_pred_default'],
            model_name,
            save_path=f"{RESULTS_DIR}/{model_name}_confusion_matrix.png"
        )
    
    # Generate comparison plots
    print("\n[3/4] Generating comparison plots...")
    plot_roc_curves(evaluation_results, save_path=f"{RESULTS_DIR}/roc_curves_comparison.png")
    plot_pr_curves(evaluation_results, save_path=f"{RESULTS_DIR}/pr_curves_comparison.png")
    
    # Print evaluation report
    print("\n[4/4] Generating evaluation report...")
    print_evaluation_report(evaluation_results)
    
    # Save results
    results_summary = {}
    for model_name, metrics in evaluation_results.items():
        results_summary[model_name] = {
            'auroc': float(metrics['auroc']),
            'auprc': float(metrics['auprc']),
            'threshold_metrics': {
                str(k): {
                    'sensitivity': float(v['sensitivity']),
                    'specificity': float(v['specificity']),
                    'precision': float(v['precision']),
                    'f1': float(v['f1'])
                }
                for k, v in metrics['threshold_metrics'].items()
            }
        }
    
    with open(f"{RESULTS_DIR}/evaluation_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*80)
    print(f"Evaluation complete! Results saved to: {RESULTS_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()