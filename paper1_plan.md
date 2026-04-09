# Paper 1 — Plan
## Stacking Ensemble Architecture for Binary Chest X-Ray Triage on NIH ChestX-ray14

---

## Core Novelty Statement ⭐
> *"We propose a stacking ensemble framework that combines heterogeneous vision model predictions with patient metadata via a learned meta-learner, and demonstrate that this architecture outperforms single-model baselines on binary chest X-ray triage — a clinically critical screening task"*

**Rule:** Never lead with the task. Always lead with the architecture.
The task (binary chest X-ray classification) is known. The architecture is the contribution.

---

## Target Venue
- **Primary:** IEEE Journal of Biomedical and Health Informatics (IEEE JBHI)
- **Preprint:** arXiv (submit immediately on completion — establishes priority)
- **Backup:** MIDL 2025, PLOS ONE

---

## Dataset
- **NIH ChestX-ray14**
- 112,120 PNG images, 30,805 unique patients
- Source: NIH Clinical Center, USA
- Labels: 14 diseases via NLP from radiology reports (>90% accuracy)
- Binary mapping: `No Finding → 0`, `any disease present → 1`
- Class distribution: ~53.9% negative / ~46.1% positive
- Split: **Patient-level stratified** (critical — prevents leakage across follow-up scans)
  - Train: ~70% patients
  - Val: ~15% patients
  - Test: ~15% patients
- Citation: Wang et al., IEEE CVPR 2017

---

## Architecture

```
NIH ChestX-ray14 PNG Images
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              Vision Model Layer (PyTorch)            │
│                                                     │
│  EfficientNet-B3   ConvNeXt-Small   Swin-Tiny       │
│  DenseNet-121      DeiT-Small       ViT-Base        │
│                                                     │
│  • Fine-tuned end-to-end                            │
│  • Weighted BCE loss                                │
│  • Early stopping on validation MCC                 │
│  • 224×224 input for all models                    │
└──────────────────────────┬──────────────────────────┘
                           │  predicted probabilities (OOF)
                           ▼
┌─────────────────────────────────────────────────────┐
│            CatBoost Meta-Learner                    │
│                                                     │
│  Features: [prob_efficientnet, prob_convnext,       │
│             prob_swin, prob_densenet, prob_deit,    │
│             prob_vit, patient_age, patient_sex]     │
│                                                     │
│  • Trained on 3-fold OOF predictions                │
│  • auto_class_weights Balanced                      │
│  • MCC-optimised threshold                          │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
                  Final Binary Prediction
```

---

## Paper Outline (IEEE JBHI format, ~6,000 words)

### Abstract (250 words max)
- Purpose: binary chest X-ray triage using stacking ensemble
- Methods: 5 timm models + ViT + CatBoost + metadata + OOF
- Results: MCC [X] — outperforms single-model baselines
- Conclusion: stacking with OOF + metadata improves chest X-ray triage

### 1. Introduction (~600 words)
- Clinical motivation: radiologist shortage, triage need
- Limitations of single-model approaches
- Explicit contributions bullet list (IEEE expects this)

### 2. Related Work (~800 words) — 3 paragraphs structure
- Para 1: Single backbone approaches — CheXNet, DenseNet, ViT. Gap: no ensemble diversity
- Para 2: Ensemble methods in medical imaging — averaging/voting only. Gap: no learned meta-learner, no OOF, no metadata
- Para 3: THE GAP — no prior work combines diverse backbone stacking + OOF + tabular metadata in a learned meta-learner for chest X-ray binary triage

### 3. Dataset & Preprocessing (~500 words)
- NIH ChestX-ray14 description
- Binary label mapping justification
- PNG loading, normalisation, augmentation
- Patient-level split rationale and implementation
- Class distribution analysis

### 4. Methodology (~1,500 words) ← NOVELTY LIVES HERE
- 4.1 Architecture overview + diagram
- 4.2 Vision model layer — model selection rationale (diversity principle), fine-tuning strategy
- 4.3 OOF framework — why naive stacking causes leakage + diagram showing solution
- 4.4 CatBoost meta-learner — feature matrix, why metadata here not in vision models
- 4.5 MCC-optimised threshold — clinical justification

### 5. Experiments & Results (~1,000 words)
- Implementation details (hardware, versions, GitHub link)
- Baseline comparison table (CNN → each timm model → CatBoost stack)
- Ablation study:
  - Stack without metadata
  - Stack without OOF (shows leakage effect)
  - Simple averaging vs learned CatBoost stack
- Threshold analysis (MCC vs threshold curve)
- Feature importance plot

### 6. Discussion (~600 words)
- Why stacking outperforms individual models
- Metadata signal contribution
- Limitations: NLP label noise (~10%), no bounding box validation, no prospective study
- Clinical applicability: screening/triage framing, NOT diagnostic replacement
- Future work: multi-label (14 classes), Grad-CAM explainability, prospective trial

### 7. Conclusion (~200 words)

### References (~30 citations)
Key citations to include:
- Wang et al. ChestX-ray8, IEEE CVPR 2017
- Rajpurkar et al. CheXNet, arXiv 2017
- Dosovitskiy et al. ViT, ICLR 2021
- Wightman timm library, 2021
- Prokhorenkova et al. CatBoost, NeurIPS 2018
- Wolpert stacking, Neural Networks 1992
- Chicco & Jurman MCC, BioData Mining 2023

---

## Results Table (fill in after training)

| Model | AUC-ROC | AUC-PR | MCC |
|-------|---------|--------|-----|
| DenseNet-121 (CheXNet baseline) | — | — | — |
| EfficientNet-B3 | — | — | — |
| ConvNeXt-Small | — | — | — |
| Swin-Tiny | — | — | — |
| DeiT-Small | — | — | — |
| ViT-Base | — | — | — |
| **CatBoost Stack (final)** | — | — | **—** |

---

## Novelty Protection — Three Risks & Defences

1. **Risk: Prior stacking on medical images**
   → Confirm in literature review that prior ensembles use averaging/voting only — not learned stacking with OOF + metadata

2. **Risk: Small MCC improvement over single models**
   → Ablation must show each component contributes. OOF vs no-OOF. With vs without metadata. Stack vs averaging.

3. **Risk: Binary task too simple**
   → Frame as clinically necessary first-pass screening. Multi-label = future work.

---

## Code Repository
- GitHub: chest-xray-classification
- Dataset: NIH ChestX-ray14 (Kaggle + NIH Box)
- All models, OOF predictions, threshold saved and documented

---

## Timeline
| Task | Estimated Time |
|------|---------------|
| Run full pipeline on ChestX-ray14 | 1–2 weeks (compute) |
| Write Methods + Dataset sections | 1 week |
| Literature review (20–30 papers) | 2–3 weeks |
| Ablation experiments | 1 week |
| Write remaining sections | 1 week |
| Post to arXiv | Day 1 of completion |
| Submit to IEEE JBHI | Same week as arXiv |
| **Total** | **~6–8 weeks** |

---

## EB-1 Relevance
- Peer-reviewed publication in IEEE JBHI satisfies "authorship of scholarly articles"
- arXiv preprint establishes timestamp for priority
- Once published: request to review for same journal → satisfies "peer reviewer" criterion
- If cited by others: satisfies "citations" evidence
