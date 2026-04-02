"""
train_vit.py — Fine-tune google/vit-base-patch16-224-in21k for binary
               chest X-ray classification, fully in PyTorch.

Architecture
  HuggingFace ViTModel (backbone, outputs CLS token)
   Linear(hidden_size, 1)   (binary logit)

The ViT backbone is fine-tuned end-to-end with a lower learning rate
than the head to avoid destroying pretrained representations.

Usage
    python train_vit.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import ViTModel
from sklearn.metrics import matthews_corrcoef

from config import (
    VIT_MODEL_NAME, SAVE_DIR, SEED,
    MAX_EPOCHS, WEIGHT_DECAY,
    EARLY_STOP_PATIENCE, VIT_LEARNING_RATE,
)
from dataset import (
    load_training_df, train_val_split,
    make_loaders, compute_pos_weight,
)
from utils import (
    set_seed, get_device,
    find_best_threshold, compute_metrics,
    plot_training_history, save_checkpoint,
)


#  ViT wrapper 

class ViTClassifier(nn.Module):
    """
    Thin wrapper around HuggingFace ViTModel for binary classification.

    Extracts the CLS-token representation from the last hidden state
    and passes it through a single linear layer to produce a logit.

    The backbone is fine-tuned end-to-end (all parameters trainable).
    """

    def __init__(self, model_name: str = VIT_MODEL_NAME):
        super().__init__()
        self.vit     = ViTModel.from_pretrained(model_name)
        hidden_size  = self.vit.config.hidden_size    # 768 for base
        self.head    = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        pixel_values : (B, 3, H, W)  — already normalised to ImageNet stats

        Returns
        logits : (B,)
        """
        outputs    = self.vit(pixel_values=pixel_values)
        cls_token  = outputs.last_hidden_state[:, 0, :]   # (B, hidden_size)
        logits     = self.head(cls_token).squeeze(1)       # (B,)
        return logits


#  One epoch 

def train_one_epoch(model: nn.Module,
                    loader,
                    optimizer,
                    criterion,
                    device: torch.device,
                    scaler) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch in loader:
        # HuggingFace ViT expects pixel_values in channel-first format (B, 3, H, W)
        # Our DataLoader already produces tensors in that format.
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        # Gradient clipping stabilises ViT fine-tuning
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss   = total_loss / len(loader.dataset)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    _, mcc     = find_best_threshold(all_labels, all_probs)
    return avg_loss, mcc, all_probs, all_labels


#  Training loop 

def train_vit(train_df,
              val_df,
              device: torch.device,
              pos_weight: torch.Tensor) -> str:
    """
    Full training loop for the ViT classifier.

    Returns
    checkpoint_path : str
    """
    print(f"\n{'#'*60}")
    print(f"  Training: {VIT_MODEL_NAME}  (PyTorch)")
    print(f"{'#'*60}")

    set_seed(SEED)

    train_loader, val_loader = make_loaders(train_df, val_df)

    model     = ViTClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # Differential learning rates:
    #   backbone - VIT_LEARNING_RATE (small, protect pretrained weights)
    #   head     - 10× larger
    backbone_params = list(model.vit.parameters())
    head_params     = list(model.head.parameters())

    optimizer = AdamW([
        {"params": backbone_params, "lr": VIT_LEARNING_RATE},
        {"params": head_params,     "lr": VIT_LEARNING_RATE * 10},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-7)
    scaler    = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history = {"train_loss": [], "val_loss": [],
               "train_mcc":  [], "val_mcc":  []}

    best_val_mcc     = -1.0
    patience_counter = 0
    checkpoint_path  = os.path.join(SAVE_DIR, "vit_best.pt")

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

        if val_mcc > best_val_mcc:
            best_val_mcc     = val_mcc
            patience_counter = 0
            save_checkpoint(model, checkpoint_path, epoch, val_mcc,
                            extra={"model_name": "vit"})
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {EARLY_STOP_PATIENCE} epochs).")
                break

    # Final eval with best weights
    model.load_state_dict(torch.load(checkpoint_path,
                                     map_location=device)["state_dict"])
    _, _, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
    best_thresh, _ = find_best_threshold(val_labels, val_probs)
    val_preds       = (val_probs >= best_thresh).astype(int)
    compute_metrics(val_labels, val_preds, val_probs,
                    threshold_label=f"MCC-optimal ({best_thresh:.3f})")

    plot_path = os.path.join(SAVE_DIR, "vit_history.png")
    plot_training_history(history, model_name="ViT", save_path=plot_path)

    print(f"\n  Best val MCC: {best_val_mcc:.4f}")
    return checkpoint_path


#  Entry point 

def main():
    device = get_device()

    df              = load_training_df()
    train_df, val_df = train_val_split(df)
    pos_weight      = compute_pos_weight(train_df)

    train_vit(train_df, val_df, device, pos_weight)


if __name__ == "__main__":
    main()
