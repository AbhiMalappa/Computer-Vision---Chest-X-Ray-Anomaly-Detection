"""
utils.py — Shared helpers used across the pipeline.

Includes:
- MCC-optimised threshold search
 - Full metrics report (MCC, AUC-ROC, AUC-PR, recall, confusion matrix)
 - ROC / PR curve plotting
 - Seed setting
 - Model checkpoint helpers
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    matthews_corrcoef,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, recall_score, confusion_matrix,
    PrecisionRecallDisplay,
)


#  Reproducibility 

def set_seed(seed: int = 722) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#  MCC threshold optimisation 
def find_best_threshold(y_true: np.ndarray,
                        y_prob: np.ndarray,
                        n_points: int = 200) -> tuple[float, float]:
    """
    Search [0, 1] for the threshold that maximises MCC on the supplied labels.

    Returns
    best_threshold : float
    best_mcc       : float
    """
    thresholds  = np.linspace(0.01, 0.99, n_points)
    mcc_scores  = [
        matthews_corrcoef(y_true, (y_prob >= t).astype(int))
        for t in thresholds
    ]
    best_idx       = int(np.argmax(mcc_scores))
    best_threshold = float(thresholds[best_idx])
    best_mcc       = float(mcc_scores[best_idx])
    return best_threshold, best_mcc


#  Metrics 
def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray,
                    threshold_label: str = "default 0.5") -> dict:
    """
    Compute and print a full metrics report.

    Parameters
    y_true: ground-truth binary labels
    y_pred : predicted binary labels (already thresholded)
    y_prob : predicted probabilities for the positive class
    threshold_label : descriptive string for the print header

    Returns
    dict of metric name → value
    """
    fpr, tpr, _   = roc_curve(y_true, y_prob)
    roc_auc        = auc(fpr, tpr)
    prec, rec, _   = precision_recall_curve(y_true, y_prob)
    pr_auc         = auc(rec, prec)
    ap             = average_precision_score(y_true, y_prob, average="macro")
    acc            = accuracy_score(y_true, y_pred)
    recall_bin     = recall_score(y_true, y_pred, average="binary")
    recall_mac     = recall_score(y_true, y_pred, average="macro")
    mcc            = matthews_corrcoef(y_true, y_pred)
    cm             = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*55}")
    print(f" Metrics — {threshold_label} threshold")
    print(f"{'='*55}")
    print(f" Confusion Matrix:\n{cm}")
    print(f" Accuracy          : {acc:.4f}")
    print(f" AUC-ROC           : {roc_auc:.4f}")
    print(f" Average Precision : {ap:.4f}")
    print(f" AUC-PR            : {pr_auc:.4f}")
    print(f" Recall (positive) : {recall_bin:.4f}")
    print(f" Recall (macro)    : {recall_mac:.4f}")
    print(f" MCC               : {mcc:.4f}")
    print(f"{'='*55}\n")

    return {
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "avg_precision": ap, "accuracy": acc,
        "recall_positive": recall_bin, "recall_macro": recall_mac,
        "mcc": mcc, "confusion_matrix": cm,
    }


#  Plotting 
def plot_roc_pr(y_true: np.ndarray,
                y_prob: np.ndarray,
                title_suffix: str = "",
                save_path: str | None = None) -> None:
    """Plot ROC curve and Precision-Recall curve side-by-side."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc      = auc(fpr, tpr)
    n            = len(y_true)
    prevalence   = np.mean(y_true)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # ROC
    ax1.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"AUC = {roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax1.set(xlim=[0, 1], ylim=[0, 1.05],
            xlabel="False Positive Rate", ylabel="True Positive Rate",
            title=f"ROC Curve  n={n}  {title_suffix}")
    ax1.legend(loc="lower right")

    # PR (manual)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc        = auc(rec, prec)
    ax2.plot(rec, prec, color="darkorange", lw=2,
             label=f"AUC = {pr_auc:.4f}")
    ax2.plot([0, 1], [prevalence, prevalence],
             color="navy", lw=1.5, linestyle="--", label="Chance")
    ax2.set(xlim=[0, 1], ylim=[0, 1.05],
            xlabel="Recall", ylabel="Precision",
            title=f"PR Curve  n={n}  {title_suffix}")
    ax2.legend(loc="upper right")

    # PR (sklearn display)
    PrecisionRecallDisplay.from_predictions(
        y_true, y_prob, plot_chance_level=True, ax=ax3
    )
    ax3.set_title("Precision-Recall (sklearn)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    plt.show()


def plot_training_history(history: dict,
                          model_name: str = "",
                          save_path: str | None = None) -> None:
    """
    Plot training vs validation loss and MCC over epochs.
    history keys expected: 'train_loss', 'val_loss', 'train_mcc', 'val_mcc'
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set(xlabel="Epoch", ylabel="Loss",
            title=f"{model_name} — Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_mcc"], label="Train MCC")
    ax2.plot(epochs, history["val_mcc"],   label="Val MCC")
    ax2.set(xlabel="Epoch", ylabel="MCC",
            title=f"{model_name} — MCC")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    plt.show()


#  Checkpoint helpers 
def save_checkpoint(model: torch.nn.Module,
                    path: str,
                    epoch: int,
                    val_mcc: float,
                    extra: dict | None = None) -> None:
    """Save model weights and metadata to disk."""
    payload = {
        "epoch":   epoch,
        "val_mcc": val_mcc,
        "state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"  Checkpoint saved → {path}  (epoch {epoch}, val_MCC {val_mcc:.4f})")


def load_checkpoint(model: torch.nn.Module,
                    path: str,
                    device: torch.device) -> dict:
    """Load weights into model and return the checkpoint metadata dict."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"  Loaded checkpoint: {path}  "
          f"(epoch {checkpoint['epoch']}, val_MCC {checkpoint['val_mcc']:.4f})")
    return checkpoint


#  Device helper 
def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    return device
