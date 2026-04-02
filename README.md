# Chest X-Ray Anomaly Detection
### Binary Classification using Ensemble of Vision Models + CatBoost Meta-Learner



## Overview

This project is a production-grade chest X-ray binary classification pipeline:

- **Class 0** - No Finding (normal chest X-ray) 
- **Class 1** - Any abnormal finding (consolidation of 14 pathology classes)

The final classification model is a stacked ensemble: 
1.  five fine-tuned `timm` vision backbones +
2.  a fine-tuned `ViT` (Vision Transformer) +
3.  predicted probabilities from 1 and 2, combined with patient metadata are fed into a `CatBoost` meta-learner for the final prediction.

**Final MCC score: 0.91**



## Architecture

```
DICOM Images
    │
    │
─────────────────────────────────────────────────────
              Vision Model Layer (PyTorch)            
                                                     
EfficientNet-B3, ConvNeXt-Small, Swin-Tiny       
DenseNet-121, DeiT-Small, ViT-Base        
                                                     
 • Fine-tuned end-to-end on chest X-ray data        
 • Weighted BCE loss (handles 70/30 imbalance)      
 • Early stopping on validation MCC  

─────────────────────────────────────────────────────
    │  
    │  predicted probabilities   
    │                      
─────────────────────────────────────────────────────
           CatBoost Meta-Learner                    
                                                     
Features: [prob_model_1, …, prob_vit,patient_age, patient_sex]                                                            
 • Trained on Out-of-Fold (OOF) predictions to prevent leakage                               
 • MCC-optimised classification threshold       

────────────────────────────────────────────────────
    │
    │                           
Final Binary Prediction
```



## Dataset

This project uses the VinBigData Chest X-ray Abnormalities Detection dataset.

- Competition: https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection
- Organized by: Vingroup Big Data Institute

> The dataset is not included in this repository. To reproduce, use the dataset from Kaggle.


## Results

Mesured metric -- **MCC** - CatBoost Stack (final) score is 0.91

Why MCC?

Accuracy is misleading on imbalanced datasets. Matthews Correlation Coefficient (MCC) 
is a single balanced metric that accounts for all four quadrants of the confusion matrix 
(TP, TN, FP, FN). It is the primary evaluation metric for this competition and for 
clinical relevance.

## Project Structure

```
chest_xray/
│
├── config.py             # All hyperparameters, paths, model names
├── dataset.py            # PyTorch Dataset: DICOM loading, augmentation, metadata
├── utils.py              # MCC threshold search, metrics, plotting, checkpointing
│
├── train_timm.py         # Fine-tune timm vision backbones
├── train_vit.py          # Fine-tune HuggingFace ViT (PyTorch)
├── generate_oof.py       # Out-of-Fold predictions (leakage-free stacking)
├── train_catboost.py     # CatBoost meta-learner on OOF probs + metadata
├── predict.py            # Full test-set inference  submission CSV
│
├── requirements.txt
├── .gitignore
└── README.md
```


## Running the Pipeline

Fine-tune all 5 timm models -
python train_timm.py

Fine-tune ViT - 
python train_vit.py

Generate OOF predictions (prevents leakage into CatBoost)
python generate_oof.py

Train CatBoost meta-learner
python train_catboost.py

Generate test predictions + submission CSV
python predict.py


To generate OOF for a single model:
generate_oof.py --model vit


## Key Design Decisions

**Why OOF for CatBoost training?**  
If CatBoost trains on the same predictions the vision models made on their own 
training data, it learns from overfit signals. OOF generation ensures every 
prediction CatBoost trains on was made by a model that never saw that example.

**Why include patient metadata in CatBoost, not in vision models?**  
Age and sex are tabular features — tree models handle them natively and excel at 
learning interactions with predicted probabilities. Injecting metadata into vision 
model heads adds architectural complexity and can destabilise fine-tuning.

**Why weighted BCE + CatBoost class balancing?**  
The dataset is 70/30. Weighted loss at the vision model level and 
`auto_class_weights='Balanced'` at the CatBoost level means both layers of the 
stack are imbalance-aware. The MCC-optimal threshold search at the end provides 
a final correction.

**Why no vertical flip in data augmentation?**  
Chest X-rays have a fixed anatomical orientation (top-to-bottom). Vertical 
flipping creates anatomically impossible images and would hurt model learning.

