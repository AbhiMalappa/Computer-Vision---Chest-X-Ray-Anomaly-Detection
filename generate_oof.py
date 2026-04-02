"""
generate_oof.py — Out-of-Fold (OOF) predictions for all vision models (timm and VIT models).

Why OOF?
When training CatBoost on top of vision model predictions, using the same data that those models were trained on causes leakage 
— CatBoost learns from overfit predictions. OOF generation fixes this.


Outputs (saved to OOF_DIR)
  oof_<model_name>.npy - shape (N_train,)  predicted probabilities
  oof_labels.npy - shape (N_train,)  ground-truth labels
  oof_ids.npy - shape (N_train,)  image IDs (for alignment)
  oof_metadata.csv -  age + sex for every training example

Usage
    python generate_oof.py                          # all models
    python generate_oof.py --model efficientnet_b3  # one timm model
    python generate_oof.py --model vit              # ViT only
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm

from config import (
    TIMM_MODELS, OOF_DIR, SAVE_DIR, SEED,
    MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOP_PATIENCE, VIT_LEARNING_RATE, N_FOLDS,
)
from dataset import (
    load_training_df, get_kfold_splits,
    make_loaders, compute_pos_weight,
    build_metadata_df, TRAIN_DIR,
)
from train_timm import build_timm_model, train_one_epoch, evaluate as timm_evaluate
from train_vit  import ViTClassifier, train_one_epoch as vit_train_epoch
from train_vit  import evaluate as vit_evaluate
from utils import (
    set_seed, get_device,
    find_best_threshold, save_checkpoint,
)


# Generic fold trainer 

def _train_fold(model: nn.Module,
                train_df,
                val_df,
                pos_weight: torch.Tensor,
                device: torch.device,
                is_vit: bool = False,
                fold: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Train a model on one fold and return (val_probs, val_labels).
    Parameters
    model       : freshly initialised model (untrained)
    train_df    : training portion of this fold
    val_df      : held-out portion of this fold
    pos_weight  : class-imbalance weight
    device      : torch device
    is_vit      : use ViT-specific training (gradient clipping, lower LR)
    fold        : fold index (for logging)

    Returns
    val_probs  : np.ndarray, predicted probabilities for val examples
    val_labels : np.ndarray, ground-truth labels for val examples
    """
    print(f"\n    --- Fold {fold + 1}/{N_FOLDS} ---")

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    train_loader, val_loader = make_loaders(train_df, val_df)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    if is_vit:
        optimizer = AdamW([
            {"params": list(model.vit.parameters()),  "lr": VIT_LEARNING_RATE},
            {"params": list(model.head.parameters()), "lr": VIT_LEARNING_RATE * 10},
        ], weight_decay=WEIGHT_DECAY)
        _train_epoch = vit_train_epoch
        _evaluate    = vit_evaluate
    else:
        optimizer    = AdamW(model.parameters(),
                             lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        _train_epoch = train_one_epoch
        _evaluate    = timm_evaluate

    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    best_val_mcc     = -1.0
    patience_counter = 0
    best_probs       = None
    best_labels      = None

    # Temporary in-memory best state (no disk write per fold to save space)
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_mcc = _train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_mcc, val_probs, val_labels = _evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        print(f"      Epoch {epoch:3d} | "
              f"train_mcc={train_mcc:.4f}  val_mcc={val_mcc:.4f}")

        if val_mcc > best_val_mcc:
            best_val_mcc     = val_mcc
            patience_counter = 0
            best_probs       = val_probs.copy()
            best_labels      = val_labels.copy()
            best_state       = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"      Early stop (no improvement for "
                      f"{EARLY_STOP_PATIENCE} epochs).")
                break

    print(f"    Fold {fold + 1} best val MCC: {best_val_mcc:.4f}")

    # Restore best weights (for the fold's prediction)
    model.load_state_dict(best_state)
    return best_probs, best_labels


#  OOF for one model 

def generate_oof_for_model(model_name: str,
                           df,
                           device: torch.device) -> np.ndarray:
    """
    Run full K-fold OOF generation for one model.
    Returns
    oof_probs : np.ndarray of shape (len(df),) aligned to df index order
    """
    print(f"\n{'='*60}")
    print(f"  OOF generation: {model_name}  ({N_FOLDS} folds)")
    print(f"{'='*60}")

    set_seed(SEED)
    is_vit   = (model_name == "vit")
    splits   = get_kfold_splits(df, n_folds=N_FOLDS)
    oof_probs = np.full(len(df), np.nan, dtype=np.float32)

    for fold_idx, (train_df, val_df) in enumerate(splits):
        # Fresh model every fold
        if is_vit:
            model = ViTClassifier()
        else:
            model = build_timm_model(model_name)

        pos_weight = compute_pos_weight(train_df)

        val_probs, val_labels = _train_fold(
            model, train_df, val_df,
            pos_weight, device,
            is_vit=is_vit, fold=fold_idx,
        )

        # Map predictions back to the original df positions
        val_indices = splits[fold_idx][1].index.tolist()
        oof_probs[val_indices] = val_probs

    assert not np.any(np.isnan(oof_probs)), \
        "OOF array has NaNs — some indices were not filled."

    # Save
    save_name = "vit" if is_vit else model_name
    out_path  = os.path.join(OOF_DIR, f"oof_{save_name}.npy")
    np.save(out_path, oof_probs)
    print(f"\n  OOF saved → {out_path}")

    return oof_probs


#  Save shared labels + IDs + metadata once 

def save_oof_labels_and_metadata(df) -> None:
    """
    Save ground-truth labels, image IDs, and DICOM metadata to OOF_DIR.
    Only needs to run once — all model OOF arrays share the same order.
    """
    labels_path = os.path.join(OOF_DIR, "oof_labels.npy")
    ids_path    = os.path.join(OOF_DIR, "oof_ids.npy")
    meta_path   = os.path.join(OOF_DIR, "oof_metadata.csv")

    if not os.path.exists(labels_path):
        np.save(labels_path, df["Finding"].values.astype(int))
        print(f"  Labels saved  → {labels_path}")

    if not os.path.exists(ids_path):
        np.save(ids_path, df["id"].values)
        print(f"  IDs saved     → {ids_path}")

    if not os.path.exists(meta_path):
        print("  Extracting metadata from DICOMs (one-time, may take a moment)…")
        meta_df = build_metadata_df(df["id"].tolist(), TRAIN_DIR)
        # Re-align to df order
        meta_df = meta_df.loc[df["id"]]
        meta_df.to_csv(meta_path)
        print(f"  Metadata saved → {meta_path}")


#  Entry point 

def main():
    parser = argparse.ArgumentParser(
        description="Generate OOF predictions for stacking."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "Single model to generate OOF for. "
            "Use 'vit' for ViT, or a timm model name. "
            "Default: generate for all models."
        ),
    )
    args = parser.parse_args()

    device = get_device()
    df     = load_training_df()

    # Save shared metadata once
    save_oof_labels_and_metadata(df)

    all_models = TIMM_MODELS + ["vit"]
    models_to_run = [args.model] if args.model else all_models

    for model_name in models_to_run:
        generate_oof_for_model(model_name, df, device)

    print("\n" + "="*60)
    print("  All OOF predictions generated.")
    print(f"  Files in {OOF_DIR}:")
    for f in sorted(os.listdir(OOF_DIR)):
        print(f"    {f}")
    print("="*60)


if __name__ == "__main__":
    main()
