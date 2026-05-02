"""
train_catboost.py — Train CatBoost meta-learner on OOF vision model predictions
                    + patient metadata (age, sex).

Inputs (from OOF_DIR)
  oof_<model>.npy       — predicted probabilities from each vision model
  oof_labels.npy        — ground-truth labels
  oof_metadata.csv      — patient_age, patient_sex (encoded)

Feature matrix (one row per training image)
  [prob_efficientnet_b3,
   prob_convnext_small,
   prob_swin_tiny,
   prob_densenet121,
   prob_deit_small,
   patient_age,
   patient_sex]

Outputs
  saved_models/catboost_model.cbm             — trained CatBoost model
  saved_models/catboost_threshold.npy         — MCC-optimal threshold (scalar)
  saved_models/catboost_feature_importance.png
  saved_models/catboost_oof_metrics.csv       — OOF MCC summary
  saved_models/ablation_results.csv           — ablation comparison (--ablation only)
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

from config import (
    OOF_DIR, SAVE_DIR, SEED, CATBOOST_PARAMS,
    TIMM_MODELS, N_FOLDS, THRESHOLD_GRID_SIZE, SKIP_VIT,
)
from utils import find_best_threshold, compute_metrics, plot_roc_pr


#  Feature assembly

MODEL_NAMES = TIMM_MODELS if SKIP_VIT else TIMM_MODELS + ["vit"]

FEATURE_NAMES = [f"prob_{m}" for m in MODEL_NAMES] + \
                ["patient_age", "patient_sex"]


def load_oof_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and assemble the OOF feature matrix and label array.
    Returns
    X_full : np.ndarray, shape (N, n_model_probs + 2)  — probs + age + sex
    X_probs: np.ndarray, shape (N, n_model_probs)       — probs only
    y      : np.ndarray, shape (N,)
    feat_names_full : list[str]
    """
    y = np.load(os.path.join(OOF_DIR, "oof_labels.npy")).astype(int)

    prob_cols = []
    for model_name in MODEL_NAMES:
        path = os.path.join(OOF_DIR, f"oof_{model_name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"OOF file not found: {path}\n"
                f"Run generate_oof.py --model {model_name} first."
            )
        prob_cols.append(np.load(path).astype(np.float32))

    meta_path = os.path.join(OOF_DIR, "oof_metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Metadata file not found: {meta_path}\n"
            "Run generate_oof.py first."
        )
    meta_df = pd.read_csv(meta_path, index_col="image_id")
    ids     = np.load(os.path.join(OOF_DIR, "oof_ids.npy"), allow_pickle=True)
    meta_df = meta_df.loc[ids]

    age = meta_df["patient_age"].values.astype(np.float32)
    sex = meta_df["patient_sex"].values.astype(np.float32)

    X_probs = np.column_stack(prob_cols)
    X_full  = np.column_stack(prob_cols + [age, sex])

    print(f"  Feature matrix (full): {X_full.shape}  "
          f"({len(MODEL_NAMES)} model probs + 2 metadata features)")
    print(f"  Label distribution: "
          f"neg={int((y==0).sum())}  pos={int((y==1).sum())}")
    return X_full, X_probs, y


#  CatBoost training

def train_catboost(X: np.ndarray,
                   y: np.ndarray,
                   feature_names: list) -> CatBoostClassifier:
    """Train CatBoost with early stopping. Returns trained model."""
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    train_pool = Pool(X_train, y_train, feature_names=feature_names)
    val_pool   = Pool(X_val,   y_val,   feature_names=feature_names)
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model


#  Cross-validated MCC on OOF

def cross_validated_catboost_mcc(X: np.ndarray,
                                  y: np.ndarray,
                                  feature_names: list) -> tuple[float, float]:
    """3-fold CV MCC estimate. Returns (mean_mcc, std_mcc)."""
    skf  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    mccs = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_vl = X[tr_idx], X[vl_idx]
        y_tr, y_vl = y[tr_idx], y[vl_idx]

        model   = CatBoostClassifier(**{**CATBOOST_PARAMS, "verbose": 0})
        tr_pool = Pool(X_tr, y_tr, feature_names=feature_names)
        vl_pool = Pool(X_vl, y_vl, feature_names=feature_names)
        model.fit(tr_pool, eval_set=vl_pool, use_best_model=True)

        probs       = model.predict_proba(X_vl)[:, 1]
        thresh, mcc = find_best_threshold(y_vl, probs, n_points=THRESHOLD_GRID_SIZE)
        mccs.append(mcc)
        print(f"    Fold {fold}: MCC={mcc:.4f}  threshold={thresh:.3f}")

    mean_mcc = float(np.mean(mccs))
    std_mcc  = float(np.std(mccs))
    print(f"\n  CV MCC: {mean_mcc:.4f} ± {std_mcc:.4f}")
    return mean_mcc, std_mcc


#  Feature importance plot

def plot_feature_importance(model: CatBoostClassifier,
                            feature_names: list,
                            save_path: str) -> None:
    importance = model.get_feature_importance()
    sorted_idx = np.argsort(importance)
    fig, ax    = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx],
            color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title("CatBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Feature importance plot saved → {save_path}")
    plt.close()


#  Single configuration runner

def run_configuration(X: np.ndarray,
                      y: np.ndarray,
                      feature_names: list,
                      label: str) -> dict:
    """Train and evaluate one CatBoost configuration. Returns metrics dict."""
    print(f"\n{'='*60}")
    print(f"  Configuration: {label}")
    print(f"{'='*60}")

    print(f"\n  Cross-validated MCC ({N_FOLDS}-fold):")
    mean_mcc, std_mcc = cross_validated_catboost_mcc(X, y, feature_names)

    print(f"\n  Training final model…")
    model = train_catboost(X, y, feature_names)

    probs = model.predict_proba(X)[:, 1]
    best_thresh, best_mcc = find_best_threshold(y, probs, n_points=THRESHOLD_GRID_SIZE)
    preds = (probs >= best_thresh).astype(int)
    metrics = compute_metrics(y, preds, probs,
                              threshold_label=f"{label} MCC-optimal ({best_thresh:.3f})")

    return {
        "configuration": label,
        "cv_mcc_mean":   round(mean_mcc, 4),
        "cv_mcc_std":    round(std_mcc, 4),
        "oof_mcc":       round(best_mcc, 4),
        "roc_auc":       round(metrics["roc_auc"], 4),
        "pr_auc":        round(metrics["pr_auc"], 4),
        "accuracy":      round(metrics["accuracy"], 4),
        "recall_pos":    round(metrics["recall_positive"], 4),
        "threshold":     round(best_thresh, 4),
        "_model":        model,
        "_thresh":       best_thresh,
    }


#  Entry point

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation: models-only vs models+metadata")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  CatBoost Meta-Learner Training")
    print("="*60)

    X_full, X_probs, y = load_oof_features()
    prob_feature_names  = [f"prob_{m}" for m in MODEL_NAMES]

    if args.ablation:
        # ── Ablation: three configurations ────────────────────────────────
        rows = []

        # 1. Simple average ensemble (no CatBoost)
        print(f"\n{'='*60}")
        print(f"  Configuration: Simple average (no CatBoost)")
        print(f"{'='*60}")
        avg_probs = X_probs.mean(axis=1)
        avg_thresh, avg_mcc = find_best_threshold(y, avg_probs, n_points=THRESHOLD_GRID_SIZE)
        avg_preds   = (avg_probs >= avg_thresh).astype(int)
        avg_metrics = compute_metrics(y, avg_preds, avg_probs,
                                      threshold_label=f"Simple average (thresh={avg_thresh:.3f})")
        rows.append({
            "configuration": "Simple average (no CatBoost)",
            "cv_mcc_mean":   "N/A",
            "cv_mcc_std":    "N/A",
            "oof_mcc":       round(avg_mcc, 4),
            "roc_auc":       round(avg_metrics["roc_auc"], 4),
            "pr_auc":        round(avg_metrics["pr_auc"], 4),
            "accuracy":      round(avg_metrics["accuracy"], 4),
            "recall_pos":    round(avg_metrics["recall_positive"], 4),
            "threshold":     round(avg_thresh, 4),
        })

        # 2. CatBoost — models only (no metadata)
        result = run_configuration(X_probs, y, prob_feature_names, "CatBoost — models only (no metadata)")
        rows.append({k: v for k, v in result.items() if not k.startswith("_")})

        # 3. CatBoost — models + metadata (full stack)
        full_result = run_configuration(X_full, y, FEATURE_NAMES, "CatBoost — models + metadata (age, sex)")
        rows.append({k: v for k, v in full_result.items() if not k.startswith("_")})

        ablation_df = pd.DataFrame(rows)
        ablation_path = os.path.join(SAVE_DIR, "ablation_results.csv")
        ablation_df.to_csv(ablation_path, index=False)

        print("\n" + "="*60)
        print("  ABLATION RESULTS")
        print("="*60)
        print(ablation_df.to_string(index=False))
        print(f"\n  Ablation table saved → {ablation_path}")

        model       = full_result["_model"]
        best_thresh = full_result["_thresh"]

    else:
        # ── Standard training ──────────────────────────────────────────────
        result     = run_configuration(X_full, y, FEATURE_NAMES,
                                       "Models + metadata (age, sex)")
        model      = result["_model"]
        best_thresh = result["_thresh"]
        mean_mcc   = result["cv_mcc_mean"]
        std_mcc    = result["cv_mcc_std"]
        best_mcc   = result["oof_mcc"]

        metrics_path = os.path.join(SAVE_DIR, "catboost_oof_metrics.csv")
        pd.DataFrame([{
            "cv_mcc_mean":   mean_mcc,
            "cv_mcc_std":    std_mcc,
            "oof_mcc":       best_mcc,
            "oof_threshold": round(best_thresh, 4),
        }]).to_csv(metrics_path, index=False)
        print(f"  Metrics saved → {metrics_path}")

    # Save production model + threshold
    model_path = os.path.join(SAVE_DIR, "catboost_model.cbm")
    model.save_model(model_path)
    print(f"\n  CatBoost model saved → {model_path}")

    thresh_path = os.path.join(SAVE_DIR, "catboost_threshold.npy")
    np.save(thresh_path, np.array([best_thresh]))
    print(f"  Threshold saved     → {thresh_path}")

    # ROC/PR + feature importance for the production model
    probs = model.predict_proba(X_full)[:, 1]
    plot_roc_pr(y, probs, title_suffix="CatBoost Stack (OOF)",
                save_path=os.path.join(SAVE_DIR, "catboost_roc_pr.png"))
    plot_feature_importance(model, FEATURE_NAMES,
                            save_path=os.path.join(SAVE_DIR,
                                                   "catboost_feature_importance.png"))


if __name__ == "__main__":
    main()
