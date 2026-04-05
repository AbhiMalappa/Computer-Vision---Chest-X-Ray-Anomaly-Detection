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

