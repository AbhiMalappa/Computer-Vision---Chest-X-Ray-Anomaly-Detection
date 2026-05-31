# Chest X-Ray Anomaly Detection
### Stacking Heterogeneous Vision Models for Binary Triage on NIH ChestX-ray14

Companion code for the paper:
**"Stacking Heterogeneous Vision Models Improves Chest X-Ray Triage: A Benchmark Study on NIH ChestX-ray14"**
*Submitted to IEEE Journal of Biomedical and Health Informatics*

---

## Overview

A two-layer stacking ensemble for binary chest X-ray triage on the NIH ChestX-ray14 dataset:

- **Layer 1** — Five architecturally diverse vision backbones fine-tuned independently
- **Layer 2** — CatBoost meta-learner trained on leakage-free out-of-fold (OOF) predictions + patient metadata

**Binary task:** No Finding (class 0) vs. any thoracic abnormality (class 1)

### Results (held-out test set, 17,729 images)

| Model | MCC | AUC-ROC | AUC-PR | Recall |
|---|---|---|---|---|
| EfficientNet-B3 | 0.4303 | 0.7706 | 0.7190 | 0.7169 |
| ConvNeXt-Small | 0.4178 | 0.7695 | 0.7148 | 0.6430 |
| Swin-Tiny | 0.4325 | 0.7751 | 0.7192 | 0.6929 |
| DenseNet-121 | 0.4312 | 0.7727 | 0.7231 | 0.7006 |
| DeiT-Small | 0.4156 | 0.7643 | 0.7132 | 0.6878 |
| **CatBoost Stack (proposed)** | **0.4373** | **0.7823** | **0.7298** | **0.7589** |

The stacking ensemble achieves **+6.6 percentage points recall** over the best single backbone (Swin-Tiny) — equivalent to 543 additional abnormal studies correctly identified per 17,729 screened radiographs.

---

## Dataset

**NIH ChestX-ray14** — 112,120 frontal-view chest X-ray PNG images from 30,805 unique patients.
Download from Kaggle: [NIH Chest X-rays](https://www.kaggle.com/datasets/nih-chest-xrays/data)

All data splitting is performed at the **patient level** to prevent leakage from follow-up scans:

| Subset | Patients | Images |
|---|---|---|
| Training | ~21,565 (70%) | 77,601 |
| Validation | ~4,620 (15%) | ~16,790 |
| Test | ~4,620 (15%) | ~17,729 |

---

## Pipeline

```
train_timm.py          — Fine-tune each vision backbone on the full training set
generate_oof.py        — Generate leakage-free OOF predictions (3-fold, patient-level)
train_catboost.py      — Train CatBoost meta-learner on OOF probs + patient metadata
predict.py             — Evaluate all models on the held-out test set; generate paper figures
```

### Architecture

- **5 vision backbones** via [timm](https://github.com/huggingface/pytorch-image-models):
  `efficientnet_b3`, `convnext_small`, `swin_tiny_patch4_window7_224`, `densenet121`, `deit_small_patch16_224`
- **3-fold patient-level OOF** cross-validation prevents data leakage into the meta-learner
- **CatBoost meta-learner** with 7 features: 5 OOF probabilities + patient age + patient sex
- **MCC-optimal threshold** selected by grid search over 200 candidates (τ* = 0.488)

---

## Running on Kaggle (recommended — GPU T4x2)

### 1. Setup

```bash
# Clone the repo into your Kaggle notebook
!git clone https://github.com/AbhiMalappa/Computer-Vision---Chest-X-Ray-Anomaly-Detection.git
%cd Computer-Vision---Chest-X-Ray-Anomaly-Detection

# Add the NIH Chest X-rays dataset as a Kaggle input dataset
# Settings > Add Data > Search "NIH Chest X-rays"
```

### 2. Train vision models

```bash
!python train_timm.py --model efficientnet_b3
!python train_timm.py --model convnext_small
!python train_timm.py --model swin_tiny_patch4_window7_224
!python train_timm.py --model densenet121
!python train_timm.py --model deit_small_patch16_224
```

### 3. Generate OOF predictions

```bash
!python generate_oof.py --model efficientnet_b3
!python generate_oof.py --model convnext_small
!python generate_oof.py --model swin_tiny_patch4_window7_224
!python generate_oof.py --model densenet121
!python generate_oof.py --model deit_small_patch16_224
```

### 4. Train CatBoost meta-learner

```bash
# Standard training
!python train_catboost.py

# Or with ablation study (simple average vs no-metadata vs full stack)
!python train_catboost.py --ablation
```

### 5. Evaluate on test set

```bash
!python predict.py
```

Outputs saved to `/kaggle/working/submissions/`:
- `model_comparison_<timestamp>.csv` — MCC, AUC-ROC, AUC-PR, Recall for all models
- `table6_confusion_matrix_comparison_<timestamp>.csv` — Confusion matrix at 0.5 and MCC-optimal thresholds
- `per_pathology_breakdown_<timestamp>.csv` — Per-disease detection rates
- `fig2_roc_all_models_<timestamp>.png` — ROC curves
- `fig3_pr_all_models_<timestamp>.png` — Precision-Recall curves
- `fig4_mcc_vs_threshold_<timestamp>.png` — MCC vs threshold curve

---

## Configuration

All settings are in `config.py`. Key options:

| Setting | Default | Description |
|---|---|---|
| `DEBUG_SUBSET` | `None` | Set to an integer (e.g. `500`) to subsample patients for fast testing |
| `N_FOLDS` | `3` | Number of OOF folds |
| `SEED` | `722` | Global random seed |
| `SKIP_VIT` | `True` | Skip ViT model (not included in current pipeline) |
| `MAX_EPOCHS` | `30` | Max training epochs per model |
| `EARLY_STOP_PATIENCE` | `5` | Epochs without val MCC improvement before stopping |

To override `DEBUG_SUBSET` in a Kaggle notebook without editing the file:

```python
import config
config.DEBUG_SUBSET = 500   # quick smoke test
```

---

## Requirements

```
torch>=2.1
timm>=0.9.12
catboost>=1.2.2
torchvision
scikit-learn
pandas
numpy
matplotlib
Pillow
```

---

## Citation

If you use this code, please cite:

> A. Malappa, "Stacking Heterogeneous Vision Models Improves Chest X-Ray Triage: A Benchmark Study on NIH ChestX-ray14," submitted to *IEEE Journal of Biomedical and Health Informatics*, 2024.
