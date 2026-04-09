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
   prob_vit,
   patient_age,
   patient_sex]

Outputs
  saved_models/catboost_model.cbm — trained CatBoost model
  saved_models/catboost_threshold.npy — MCC-optimal threshold (scalar)
  saved_models/catboost_feature_importance.png
"""

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


def load_oof_features() -> tuple[np.ndarray, np.ndarray]:
    """
    Load and assemble the OOF feature matrix and label array.
    Returns
    X : np.ndarray, shape (N, n_features)
    y : np.ndarray, shape (N,)
    """
    # Labels
    y = np.load(os.path.join(OOF_DIR, "oof_labels.npy")).astype(int)

    # Vision model probabilities
    prob_cols = []
    for model_name in MODEL_NAMES:
        path = os.path.join(OOF_DIR, f"oof_{model_name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"OOF file not found: {path}\n"
                f"Run generate_oof.py --model {model_name} first."
            )
        probs = np.load(path).astype(np.float32)
        prob_cols.append(probs)

    # Metadata
    meta_path = os.path.join(OOF_DIR, "oof_metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Metadata file not found: {meta_path}\n"
            "Run generate_oof.py first."
        )
    meta_df = pd.read_csv(meta_path, index_col="image_id")

    # Re-order metadata to match OOF label order
    ids_path = os.path.join(OOF_DIR, "oof_ids.npy")
    ids      = np.load(ids_path, allow_pickle=True)
    meta_df  = meta_df.loc[ids]

    age = meta_df["patient_age"].values.astype(np.float32)
    sex = meta_df["patient_sex"].values.astype(np.float32)

    # Stack into feature matrix
    X = np.column_stack(prob_cols + [age, sex])    # (N, n_features)

    print(f"  Feature matrix: {X.shape}  "
          f"({len(MODEL_NAMES)} model probs + 2 metadata features)")
    print(f"  Label distribution: "
          f"neg={int((y==0).sum())}  pos={int((y==1).sum())}")
    return X, y


#  CatBoost training ───────────────────────────────────────────────────────

def train_catboost(X: np.ndarray, y: np.ndarray) -> CatBoostClassifier:
    """
    Train CatBoost with early stopping using an internal validation split.
    Uses auto_class_weights='Balanced' to handle 70/30 imbalance.
    Returns the trained CatBoostClassifier.
    """
    # Hold out 20% for CatBoost's own early stopping eval
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2,
        stratify=y, random_state=SEED,
    )

    train_pool = Pool(X_train, y_train, feature_names=FEATURE_NAMES)
    val_pool   = Pool(X_val,   y_val,   feature_names=FEATURE_NAMES)

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
    )

    return model


#  Cross-validated MCC on OOF 

def cross_validated_catboost_mcc(X: np.ndarray,
                                 y: np.ndarray) -> tuple[float, float]:
    """
    Estimate CatBoost performance with another layer of K-fold CV on the OOF
    features. This gives an honest estimate of the full stack's performance
    without touching the test set.

    Returns
    mean_mcc : float
    std_mcc  : float
    """
    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=SEED)
    mccs    = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_vl = X[tr_idx], X[vl_idx]
        y_tr, y_vl = y[tr_idx], y[vl_idx]

        model = CatBoostClassifier(**{**CATBOOST_PARAMS, "verbose": 0})
        tr_pool = Pool(X_tr, y_tr, feature_names=FEATURE_NAMES)
        vl_pool = Pool(X_vl, y_vl, feature_names=FEATURE_NAMES)
        model.fit(tr_pool, eval_set=vl_pool, use_best_model=True)

        probs        = model.predict_proba(X_vl)[:, 1]
        thresh, mcc  = find_best_threshold(y_vl, probs,
                                           n_points=THRESHOLD_GRID_SIZE)
        mccs.append(mcc)
        print(f"    Fold {fold}: MCC={mcc:.4f}  threshold={thresh:.3f}")

    mean_mcc = float(np.mean(mccs))
    std_mcc  = float(np.std(mccs))
    print(f"\n  CV MCC: {mean_mcc:.4f} ± {std_mcc:.4f}")
    return mean_mcc, std_mcc


#  Feature importance plot 

def plot_feature_importance(model: CatBoostClassifier,
                            save_path: str) -> None:
    importance = model.get_feature_importance()
    names      = FEATURE_NAMES

    sorted_idx = np.argsort(importance)
    fig, ax    = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(names)[sorted_idx], importance[sorted_idx],
            color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title("CatBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Feature importance plot saved → {save_path}")
    plt.close()


#  Entry point 

def main():
    print("\n" + "="*60)
    print("  CatBoost Meta-Learner Training")
    print("="*60)

    # 1. Load OOF features
    X, y = load_oof_features()

    # 2. Cross-validated MCC estimate (honest performance measure)
    print("\n  Cross-validated MCC on OOF features:")
    mean_mcc, std_mcc = cross_validated_catboost_mcc(X, y)

    # 3. Train final CatBoost on ALL OOF data
    print("\n  Training final CatBoost on full OOF features…")
    model = train_catboost(X, y)

    # 4. Find MCC-optimal threshold on full OOF
    probs         = model.predict_proba(X)[:, 1]
    best_thresh, best_mcc = find_best_threshold(
        y, probs, n_points=THRESHOLD_GRID_SIZE
    )
    preds = (probs >= best_thresh).astype(int)
    print(f"\n  Final OOF threshold: {best_thresh:.4f}  MCC: {best_mcc:.4f}")
    compute_metrics(y, preds, probs,
                    threshold_label=f"MCC-optimal ({best_thresh:.3f})")

    # 5. ROC / PR curves
    plot_roc_pr(y, probs, title_suffix="CatBoost Stack (OOF)",
                save_path=os.path.join(SAVE_DIR, "catboost_roc_pr.png"))

    # 6. Feature importance
    plot_feature_importance(
        model,
        save_path=os.path.join(SAVE_DIR, "catboost_feature_importance.png"),
    )

    # 7. Save model and threshold
    model_path = os.path.join(SAVE_DIR, "catboost_model.cbm")
    model.save_model(model_path)
    print(f"\n  CatBoost model saved → {model_path}")

    thresh_path = os.path.join(SAVE_DIR, "catboost_threshold.npy")
    np.save(thresh_path, np.array([best_thresh]))
    print(f"  Threshold saved     → {thresh_path}")

    print(f"\n  Summary")
    print(f"  -------")
    print(f"  CV MCC (OOF, {N_FOLDS}-fold): {mean_mcc:.4f} ± {std_mcc:.4f}")
    print(f"  Final OOF MCC:               {best_mcc:.4f}")


if __name__ == "__main__":
    main()
