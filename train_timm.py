"""
train_timm.py — Fine-tune each timm backbone on chest X-ray binary classification.

For every model in config.TIMM_MODELS this script:
  1. Builds the model (pretrained ImageNet weights, custom 1-output head).
  2. Trains with BCEWithLogitsLoss weighted for class imbalance.
  3. Evaluates val MCC each epoch and saves the best checkpoint.
  4. Applies early stopping when val MCC stops improving.

Usage
    python train_timm.py                        # trains all models
    python train_timm.py --model efficientnet_b3  # trains one model
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from sklearn.metrics import matthews_corrcoef

from config import (
    TIMM_MODELS, SAVE_DIR, SEED,
    MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOP_PATIENCE, BATCH_SIZE,
)
from dataset import (
    load_nih_csv, patient_level_split,
    make_loaders, compute_pos_weight,
)
from utils import (
    set_seed, get_device,
    find_best_threshold, compute_metrics,
    plot_training_history, save_checkpoint,
)


#  Model factory 

def build_timm_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Create a timm model with a single sigmoid output (binary classification).

    All backbone weights are pretrained on ImageNet.
    The classifier head is replaced with Linear(features, 1).
    We return raw logits (sigmoid applied at inference / loss stage).
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=1,      # single logit output
    )
    return model


#  One epoch 

def train_one_epoch(model: nn.Module,
                    loader,
                    optimizer,
                    criterion,
                    device: torch.device,
                    scaler) -> tuple[float, float]:
    """
    Run one training epoch.

    Returns
    avg_loss : float
    mcc      : float  (computed at 0.5 threshold over all batches)
    """
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images).squeeze(1)      # (B,)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    preds    = (np.array(all_probs) >= 0.5).astype(int)
    mcc      = matthews_corrcoef(all_labels, preds)
    return avg_loss, mcc


@torch.no_grad()
def evaluate(model: nn.Module,
             loader,
             criterion,
             device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate on a validation DataLoader.

    Returns
    avg_loss  : float
    mcc       : float  (at MCC-optimal threshold)
    all_probs : np.ndarray of predicted probabilities
    all_labels: np.ndarray of ground-truth labels
    """
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images).squeeze(1)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss  = total_loss / len(loader.dataset)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Use MCC-optimal threshold for epoch MCC tracking
    _, mcc = find_best_threshold(all_labels, all_probs)
    return avg_loss, mcc, all_probs, all_labels


#  Training loop 

def train_model(model_name: str,
                train_df,
                val_df,
                device: torch.device,
                pos_weight: torch.Tensor) -> str:
    """
    Full training loop for one timm model.

    Parameters
    model_name  : timm model name string
    train_df    : training split DataFrame
    val_df      : validation split DataFrame
    device      : torch device
    pos_weight  : class-imbalance weight tensor for BCEWithLogitsLoss

    Returns
    checkpoint_path : str  — path to the best saved checkpoint
    """
    print(f"\n{'#'*60}")
    print(f"  Training: {model_name}")
    print(f"{'#'*60}")

    set_seed(SEED)

    # Data
    train_loader, val_loader = make_loaders(train_df, val_df)

    # Model
    model = build_timm_model(model_name).to(device)

    # Loss — weighted BCE handles 70/30 imbalance
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device)
    )

    # Optimiser + scheduler
    optimizer = AdamW(model.parameters(),
                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    # Mixed-precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # Training state
    history = {"train_loss": [], "val_loss": [],
               "train_mcc":  [], "val_mcc":  []}
    best_val_mcc    = -1.0
    patience_counter = 0
    checkpoint_path  = os.path.join(SAVE_DIR, f"{model_name}_best.pt")

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_mcc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_mcc, val_probs, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mcc"].append(train_mcc)
        history["val_mcc"].append(val_mcc)

        print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
              f"train_loss={train_loss:.4f}  train_mcc={train_mcc:.4f} | "
              f"val_loss={val_loss:.4f}  val_mcc={val_mcc:.4f}")

        # Checkpoint on improvement
        if val_mcc > best_val_mcc:
            best_val_mcc     = val_mcc
            patience_counter = 0
            save_checkpoint(model, checkpoint_path, epoch, val_mcc,
                            extra={"model_name": model_name})
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {EARLY_STOP_PATIENCE} epochs).")
                break

    # Final metrics on val using best checkpoint
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device)["state_dict"])
    _, _, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
    best_thresh, _ = find_best_threshold(val_labels, val_probs)
    val_preds       = (val_probs >= best_thresh).astype(int)
    compute_metrics(val_labels, val_preds, val_probs,
                    threshold_label=f"MCC-optimal ({best_thresh:.3f})")

    # Plot training curves
    plot_path = os.path.join(SAVE_DIR, f"{model_name}_history.png")
    plot_training_history(history, model_name=model_name, save_path=plot_path)

    print(f"\n  Best val MCC: {best_val_mcc:.4f}")
    return checkpoint_path


#  Entry point 

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune timm models on chest X-ray classification."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Single timm model name to train (default: train all)."
    )
    args = parser.parse_args()

    device = get_device()

    # Load data
    df = load_nih_csv()
    train_df, val_df, _ = patient_level_split(df)
    pos_weight = compute_pos_weight(train_df)

    models_to_train = [args.model] if args.model else TIMM_MODELS

    results = {}
    for model_name in models_to_train:
        ckpt_path = train_model(
            model_name, train_df, val_df, device, pos_weight
        )
        results[model_name] = ckpt_path

    print("\n" + "="*60)
    print("  Training complete. Checkpoints:")
    for name, path in results.items():
        print(f"    {name:45s} → {path}")
    print("="*60)


if __name__ == "__main__":
    main()
