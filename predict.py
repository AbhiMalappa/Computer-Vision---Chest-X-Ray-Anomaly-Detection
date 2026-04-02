"""
predict.py — Full inference pipeline on the held-out test set.

Steps
1. Load each fine-tuned timm model + ViT from their best checkpoints.
2. Run inference on test images.  collect predicted probabilities.
3. Load patient metadata from test DICOMs.
4. Assemble feature matrix [model probs…, age, sex].
5. Load trained CatBoost model + MCC-optimal threshold.
6. Produce final binary predictions.
7. Save submission CSV.

"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import timm
from catboost import CatBoostClassifier, Pool

from config import (
    TIMM_MODELS, SAVE_DIR, SUBMIT_DIR,
    BATCH_SIZE, SEED,
)
from dataset import (
    load_test_df, make_test_loader,
    build_metadata_df, TEST_DIR,
)
from train_timm import build_timm_model
from train_vit  import ViTClassifier
from train_catboost import FEATURE_NAMES, MODEL_NAMES
from utils import get_device, set_seed, load_checkpoint


#  Single-model inference 

@torch.no_grad()
def predict_probs(model: torch.nn.Module,
                  loader,
                  device: torch.device) -> np.ndarray:
    """
    Run inference with a PyTorch model and return predicted probabilities.

    Returns
    probs : np.ndarray, shape (N,)
    """
    model.eval()
    all_probs = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())

    return np.array(all_probs, dtype=np.float32)


#  Load all vision models 

def load_all_models(device: torch.device) -> dict:
    """
    Load each saved vision model checkpoint.
    Returns - models : dict  { model_name: nn.Module (eval mode, on device) }
    """
    models = {}

    # timm models
    for model_name in TIMM_MODELS:
        ckpt_path = os.path.join(SAVE_DIR, f"{model_name}_best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Run: python train_timm.py --model {model_name}"
            )
        model = build_timm_model(model_name, pretrained=False).to(device)
        load_checkpoint(model, ckpt_path, device)
        model.eval()
        models[model_name] = model

    # ViT
    vit_path = os.path.join(SAVE_DIR, "vit_best.pt")
    if not os.path.exists(vit_path):
        raise FileNotFoundError(
            f"ViT checkpoint not found: {vit_path}\n"
            "Run: python train_vit.py"
        )
    vit = ViTClassifier().to(device)
    load_checkpoint(vit, vit_path, device)
    vit.eval()
    models["vit"] = vit

    return models


#  Assemble test feature matrix 

def build_test_features(test_df: pd.DataFrame,
                        models: dict,
                        device: torch.device) -> np.ndarray:
    """
    Run each vision model on the test set and assemble the feature matrix
    [prob_model1, …, prob_vit, patient_age, patient_sex].

    Returns
    X_test : np.ndarray, shape (N_test, n_features)
    """
    test_loader = make_test_loader(test_df)
    prob_cols   = []

    for model_name in MODEL_NAMES:
        print(f"  Predicting with {model_name}…")
        model = models[model_name]
        probs = predict_probs(model, test_loader, device)
        prob_cols.append(probs)
        print(f"    → {len(probs)} predictions, "
              f"mean prob={probs.mean():.4f}")

    # Metadata
    print("  Extracting test metadata…")
    meta_df = build_metadata_df(test_df["id"].tolist(), TEST_DIR)
    # Align to test_df order
    meta_df  = meta_df.loc[test_df["id"]]
    age = meta_df["patient_age"].values.astype(np.float32)
    sex = meta_df["patient_sex"].values.astype(np.float32)

    X_test = np.column_stack(prob_cols + [age, sex])
    print(f"  Test feature matrix: {X_test.shape}")
    return X_test


#  Final prediction 

def run_catboost_inference(X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Load CatBoost model + threshold, return (probs, binary_preds).
    """
    model_path = os.path.join(SAVE_DIR, "catboost_model.cbm")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"CatBoost model not found: {model_path}\n"
            "Run: python train_catboost.py"
        )

    thresh_path = os.path.join(SAVE_DIR, "catboost_threshold.npy")
    if not os.path.exists(thresh_path):
        raise FileNotFoundError(
            f"Threshold file not found: {thresh_path}\n"
            "Run: python train_catboost.py"
        )

    cb_model   = CatBoostClassifier()
    cb_model.load_model(model_path)
    threshold  = float(np.load(thresh_path)[0])

    test_pool  = Pool(X_test, feature_names=FEATURE_NAMES)
    probs      = cb_model.predict_proba(test_pool)[:, 1]
    preds      = (probs >= threshold).astype(int)

    print(f"  CatBoost threshold (MCC-optimal): {threshold:.4f}")
    print(f"  Predicted positives: {preds.sum()} / {len(preds)}")
    return probs, preds


# Entry point 

def main():
    print("\n" + "="*60)
    print("  Full Test-Set Inference")
    print("="*60)

    set_seed(SEED)
    device   = get_device()

    # Test data
    test_df  = load_test_df()
    print(f"  Test images: {len(test_df)}")

    # Load all vision models
    print("\n  Loading vision model checkpoints…")
    models   = load_all_models(device)

    # Build feature matrix
    print("\n  Running inference on test images…")
    X_test   = build_test_features(test_df, models, device)

    # CatBoost final prediction
    print("\n  Running CatBoost meta-learner…")
    probs, preds = run_catboost_inference(X_test)

    # Submission DataFrame
    submit = pd.DataFrame({
        "id":      test_df["id"].values,
        "Outcome": preds,
    })

    # Optional: include probability column for inspection
    submit_with_prob = submit.copy()
    submit_with_prob["pred_prob"] = probs

    timestamp    = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    submit_path  = os.path.join(SUBMIT_DIR, f"submit_{timestamp}.csv")
    debug_path   = os.path.join(SUBMIT_DIR, f"submit_{timestamp}_with_probs.csv")

    submit.to_csv(submit_path, index=False)
    submit_with_prob.to_csv(debug_path, index=False)

    print(f"\n  Submission saved  → {submit_path}")
    print(f"  Debug copy saved  → {debug_path}")
    print(f"\n  Class distribution:")
    print(submit["Outcome"].value_counts().to_string())
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
