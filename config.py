"""
config.py — Central configuration for all hyperparameters, paths, and model names.
Edit this file to change any setting across the entire pipeline.
"""

import os

#Paths 
DATA_DIR        = "../data"
TRAIN_DIR       = os.path.join(DATA_DIR, "Training")
TEST_DIR        = os.path.join(DATA_DIR, "Test")
DATA_CSV        = os.path.join(DATA_DIR, "data.csv")
SAVE_DIR        = "./saved_models"
OOF_DIR         = "./oof_predictions"
SUBMIT_DIR      = "./submissions"

os.makedirs(SAVE_DIR,   exist_ok=True)
os.makedirs(OOF_DIR,    exist_ok=True)
os.makedirs(SUBMIT_DIR, exist_ok=True)

# Reproducibility
SEED = 722

# Image settings
IMG_SIZE    = 224          # All models use 224×224
NUM_CHANNELS = 3           # Grayscale → replicated to 3 channels

# Training
BATCH_SIZE      = 32
NUM_WORKERS     = 4
MAX_EPOCHS      = 30
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
EARLY_STOP_PATIENCE = 5    # epochs without val MCC improvement

# Class imbalance 
# Computed from training data: ~70% negative, ~30% positive
# pos_weight = n_neg / n_pos ≈ 2.41  (used in BCEWithLogitsLoss)
# Will be recomputed dynamically in train scripts, but set a fallback here
POS_WEIGHT_FALLBACK = 2.41

# Cross-validation
N_FOLDS = 5                # Stratified K-Fold for OOF generation

# timm models 
# Diverse backbone selection: CNN, ConvNeXt, Swin Transformer, DenseNet, DeiT
TIMM_MODELS = [
    "efficientnet_b3",
    "convnext_small",
    "swin_tiny_patch4_window7_224",
    "densenet121",
    "deit_small_patch16_224",
]

# ViT (HuggingFace)
VIT_MODEL_NAME  = "google/vit-base-patch16-224-in21k"
VIT_LEARNING_RATE = 5e-5   # ViT typically needs a smaller LR

# CatBoost
CATBOOST_PARAMS = {
    "iterations":        1000,
    "learning_rate":     0.05,
    "depth":             6,
    "loss_function":     "Logloss",
    "eval_metric":       "MCC",
    "random_seed":       SEED,
    "early_stopping_rounds": 50,
    "verbose":           100,
    "auto_class_weights": "Balanced",   # handles 70/30 imbalance
}

# Metadata columns fed into CatBoost
META_FEATURES = ["patient_age", "patient_sex"]

# MCC threshold search
THRESHOLD_GRID_SIZE = 200  # number of candidate thresholds in [0, 1]
