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
    BATCH_SIZE, SEED, SKIP_VIT,
)
from dataset import (
    load_nih_csv, patient_level_split,
    make_test_loader, build_metadata_df,
)
from train_timm import build_timm_model
from train_vit  import ViTClassifier
from train_catboost import FEATURE_NAMES, MODEL_NAMES
from utils import get_device, set_seed, load_checkpoint, find_best_threshold, compute_metrics


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
            logits = model(images).reshape(-1)   # handles (B,1) from timm and (B,) from ViT
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
    if not SKIP_VIT:
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
                        device: torch.device) -> tuple[np.ndarray, dict]:
    """
    Run each vision model on the test set and assemble the feature matrix
    [prob_model1, …, prob_vit, patient_age, patient_sex].

    Returns
    X_test    : np.ndarray, shape (N_test, n_features)
    prob_dict : dict { model_name: probs np.ndarray } for individual evaluation
    """
    test_loader = make_test_loader(test_df)
    prob_cols   = []
    prob_dict   = {}

    for model_name in MODEL_NAMES:
        print(f"  Predicting with {model_name}…")
        model = models[model_name]
        probs = predict_probs(model, test_loader, device)
        prob_cols.append(probs)
        prob_dict[model_name] = probs
        print(f"    → {len(probs)} predictions, "
              f"mean prob={probs.mean():.4f}")

    # Metadata
    print("  Extracting test metadata…")
    meta_df = build_metadata_df(test_df)
    age = meta_df["patient_age"].values.astype(np.float32)
    sex = meta_df["patient_sex"].values.astype(np.float32)

    X_test = np.column_stack(prob_cols + [age, sex])
    print(f"  Test feature matrix: {X_test.shape}")
    return X_test, prob_dict


#  Individual baseline model evaluation

def evaluate_individual_models(prob_dict: dict,
                                y_true: np.ndarray) -> pd.DataFrame:
    """
    For each vision model, find the MCC-optimal threshold on the test set
    and compute full metrics. Prints a summary table for the paper.

    Returns
    results_df : DataFrame with one row per model
    """
    print("\n" + "="*60)
    print("  Individual Model Performance on Test Set")
    print("="*60)

    rows = []
    for model_name, probs in prob_dict.items():
        thresh, mcc = find_best_threshold(y_true, probs)
        preds = (probs >= thresh).astype(int)
        metrics = compute_metrics(y_true, preds, probs,
                                  threshold_label=f"{model_name} (MCC-optimal)")
        rows.append({
            "model":     model_name,
            "test_mcc":  round(metrics["mcc"], 4),
            "roc_auc":   round(metrics["roc_auc"], 4),
            "pr_auc":    round(metrics["pr_auc"], 4),
            "threshold": round(thresh, 4),
        })

    results_df = pd.DataFrame(rows)
    print("\n  Summary — Individual Models vs Stacked Ensemble")
    print("  " + "-"*52)
    print(results_df.to_string(index=False))
    print("  " + "-"*52)
    return results_df


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

    # Test data — same patient-level split as training (fixed seed)
    df_full = load_nih_csv()
    _, _, test_df = patient_level_split(df_full)
    print(f"  Test images: {len(test_df)}")

    # Load all vision models
    print("\n  Loading vision model checkpoints…")
    models   = load_all_models(device)

    # Build feature matrix + collect per-model probabilities
    print("\n  Running inference on test images…")
    X_test, prob_dict = build_test_features(test_df, models, device)

    # Ground-truth labels for the test set
    y_true = test_df["binary_label"].values.astype(int)

    # Evaluate each individual model on the test set (baseline MCCs for paper)
    baseline_df = evaluate_individual_models(prob_dict, y_true)

    # CatBoost final prediction
    print("\n  Running CatBoost meta-learner…")
    cb_probs, preds = run_catboost_inference(X_test)

    # Stacked ensemble metrics
    thresh_path = os.path.join(SAVE_DIR, "catboost_threshold.npy")
    cb_threshold = float(np.load(thresh_path)[0])
    print("\n  Stacked Ensemble — Test Set Metrics:")
    stack_metrics = compute_metrics(y_true, preds, cb_probs,
                                    threshold_label=f"CatBoost stack (MCC-optimal {cb_threshold:.3f})")

    # Final comparison table
    print("\n" + "="*60)
    print("  PAPER RESULTS — Test Set MCC Comparison")
    print("="*60)
    for _, row in baseline_df.iterrows():
        print(f"  {row['model']:<40} MCC = {row['test_mcc']:.4f}")
    print(f"  {'Stacked Ensemble (CatBoost)':<40} MCC = {stack_metrics['mcc']:.4f}")
    print("="*60)

    # Save results table to CSV
    timestamp    = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_path = os.path.join(SUBMIT_DIR, f"model_comparison_{timestamp}.csv")
    stack_row    = pd.DataFrame([{
        "model":     "stacked_ensemble",
        "test_mcc":  round(stack_metrics["mcc"], 4),
        "roc_auc":   round(stack_metrics["roc_auc"], 4),
        "pr_auc":    round(stack_metrics["pr_auc"], 4),
        "threshold": round(cb_threshold, 4),
    }])
    pd.concat([baseline_df, stack_row], ignore_index=True).to_csv(results_path, index=False)
    print(f"\n  Results table saved → {results_path}")

    # Submission DataFrame
    submit = pd.DataFrame({
        "id":      test_df["image_id"].values,
        "Outcome": preds,
    })
    submit_with_prob = submit.copy()
    submit_with_prob["pred_prob"] = cb_probs

    submit_path = os.path.join(SUBMIT_DIR, f"submit_{timestamp}.csv")
    debug_path  = os.path.join(SUBMIT_DIR, f"submit_{timestamp}_with_probs.csv")
    submit.to_csv(submit_path, index=False)
    submit_with_prob.to_csv(debug_path, index=False)

    print(f"  Submission saved    → {submit_path}")
    print(f"  Debug copy saved    → {debug_path}")
    print(f"\n  Class distribution:")
    print(submit["Outcome"].value_counts().to_string())
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
