"""
config.py — Central configuration for NIH ChestX-ray14 pipeline (Paper 1).

Dataset : NIH ChestX-ray14
         112,120 PNG images, 30,805 unique patients
         14 disease labels → consolidated to binary
         Binary split: No Finding = 0, any disease present = 1
         Class distribution: ~53.9% negative / ~46.1% positive

Edit this file to change any setting across the entire pipeline.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = "./data"
IMAGES_DIR      = os.path.join(DATA_DIR, "images")          # PNG images folder
DATA_ENTRY_CSV  = os.path.join(DATA_DIR, "Data_Entry_2017.csv")  # labels + metadata

SAVE_DIR        = "./saved_models"
OOF_DIR         = "./oof_predictions"
SUBMIT_DIR      = "./submissions"

os.makedirs(SAVE_DIR,   exist_ok=True)
os.makedirs(OOF_DIR,    exist_ok=True)
os.makedirs(SUBMIT_DIR, exist_ok=True)

# ─── Reproducibility ──────────────────────────────────────────────────────────
SEED = 722

# ─── Image settings ───────────────────────────────────────────────────────────
IMG_SIZE     = 224      # All models use 224×224
NUM_CHANNELS = 3        # PNG grayscale → replicated to 3 channels

# ─── Binary label mapping ─────────────────────────────────────────────────────
# ChestX-ray14 Finding Labels column contains pipe-separated disease names
# e.g. "Atelectasis|Effusion" or "No Finding"
# We map: "No Finding" → 0, any disease present → 1
NO_FINDING_LABEL = "No Finding"

# ─── Patient-level split ──────────────────────────────────────────────────────
# CRITICAL: ChestX-ray14 patients have multiple follow-up images.
# Splitting at image level causes data leakage (same patient in train + test).
# We always split at PATIENT level first, then stratify by binary label.
VAL_PATIENT_FRAC  = 0.15   # 15% of patients → validation
TEST_PATIENT_FRAC = 0.15   # 15% of patients → test
# Remaining ~70% of patients → training

# ─── Debug / fast-iteration mode ──────────────────────────────────────────────
# Set to an integer to subsample that many patients for a quick end-to-end test.
# 500 patients ≈ 1,800 images — enough to verify the full pipeline in ~30 min.
# Set to None for the full 30,805-patient training run.
DEBUG_SUBSET = 500   # set to None for full run

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE           = 32
NUM_WORKERS          = 4
MAX_EPOCHS           = 30
LEARNING_RATE        = 1e-4
WEIGHT_DECAY         = 1e-4
EARLY_STOP_PATIENCE  = 5    # epochs without val MCC improvement

# ─── Class imbalance ──────────────────────────────────────────────────────────
# ChestX-ray14 binary: ~53.9% negative / ~46.1% positive — much less severe
# than VinBigData (70/30). pos_weight recomputed dynamically from training split.
# Fallback if dynamic computation fails:
POS_WEIGHT_FALLBACK = 1.17   # 53.9 / 46.1

# ─── Cross-validation (OOF) ───────────────────────────────────────────────────
# Reduced to 3 folds for ChestX-ray14 due to dataset size (112k images).
# Still patient-level stratified to prevent leakage.
N_FOLDS = 3

# ─── timm models ──────────────────────────────────────────────────────────────
TIMM_MODELS = [
    "efficientnet_b3",
    "convnext_small",
    "swin_tiny_patch4_window7_224",
    "densenet121",
    "deit_small_patch16_224",
]

# ─── ViT (HuggingFace) ────────────────────────────────────────────────────────
VIT_MODEL_NAME    = "google/vit-base-patch16-224-in21k"
VIT_LEARNING_RATE = 5e-5
SKIP_VIT          = False  # set to True to skip ViT (e.g. local CPU testing)

# ─── CatBoost ─────────────────────────────────────────────────────────────────
CATBOOST_PARAMS = {
    "iterations":            1000,
    "learning_rate":         0.05,
    "depth":                 6,
    "loss_function":         "Logloss",
    "eval_metric":           "MCC",
    "random_seed":           SEED,
    "early_stopping_rounds": 50,
    "verbose":               100,
    "auto_class_weights":    "Balanced",
}

# ─── Metadata features fed into CatBoost ─────────────────────────────────────
# ChestX-ray14 Data_Entry_2017.csv columns used:
#   Patient Age   → patient_age  (numeric)
#   Patient Gender → patient_sex (encoded: M=0, F=1)
META_FEATURES = ["patient_age", "patient_sex"]

# ─── MCC threshold search ─────────────────────────────────────────────────────
THRESHOLD_GRID_SIZE = 200
