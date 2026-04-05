# Ab Chat Scrible - Chest X-ray Classification with Ensemble Vision Models

---

## To Do List

## Target Publications (Paper 1)

- **Primary:** IEEE Journal of Biomedical and Health Informatics (IEEE JBHI)
- **Preprint:** arXiv — submit this first, same week you finish writing, regardless of where the journal paper goes
- **Backup options in order:**
  - MIDL 2025 (Medical Imaging with Deep Learning) — conference, faster turnaround
  - PLOS ONE — open access, judges on soundness not novelty, good fallback
  - Journal of Digital Imaging — lower bar, still peer-reviewed, still counts for EB-1

---


### Novelty Claim

**⭐ CORE NOVELTY STATEMENT — use this exact framing throughout the paper:**

> *"We propose a stacking ensemble framework that combines heterogeneous vision model predictions with patient metadata via a learned meta-learner, and demonstrate that this architecture outperforms single-model baselines on binary chest X-ray triage — a clinically critical screening task"*

This framing is what keeps the paper novel. The **task** (binary chest X-ray classification) is known and published. The **architecture** is the contribution. Never lead with the task — always lead with the architecture.

---

**Why this framing is protected:**

Nobody has published this specific combination:
- Diverse timm backbone ensemble (EfficientNet + ConvNeXt + Swin + DenseNet + DeiT)
- Plus ViT (HuggingFace)
- With OOF stacking to prevent leakage
- With patient metadata (age, sex) in a CatBoost meta-learner
- Evaluated with MCC-optimal thresholding
- On NIH ChestX-ray14 binary triage task

---

**Three risks to novelty — and how to defend against each:**

1. **Risk: Someone already did stacking on medical images**
   → Literature review must explicitly show prior ensembles use simple averaging/voting, NOT learned stacking with OOF + metadata. Confirm this gap.

2. **Risk: MCC improvement over single models is small**
   → Ablation study must show each component contributes measurably. Stack vs each individual model. OOF vs no-OOF. With metadata vs without metadata.

3. **Risk: Binary task seen as too simple**
   → Frame binary triage as the clinically relevant first-pass screening step. Multi-label is future work.

---

**Related Work structure that makes novelty airtight (3 paragraphs):**

- **Para 1** — Single backbone approaches (CheXNet, DenseNet, ViT). Limitation: single model, no ensemble diversity.
- **Para 2** — Ensemble methods in medical imaging. Show existing ensembles use simple averaging/voting — not learned stacking. Limitation: no meta-learner, no leakage prevention, no metadata.
- **Para 3** — The gap: no prior work combines diverse vision backbone stacking + OOF + tabular metadata in a learned meta-learner for chest X-ray binary triage. Your paper fills this gap.

---

### What the paper needs that you don't have yet

1. **Literature review** — you need to cite 20–30 existing chest X-ray AI papers and explain what your work adds
2. **Ablation study** — show each model's individual contribution to the ensemble (this is actually already easy with your pipeline)
3. **Statistical significance** — confidence intervals on your MCC, not just a point estimate
4. **Radiologist validation** — ideally one radiologist reviews a sample of predictions. This is where your Kaiser access is a huge advantage
5. **Limitations section** — distribution shift (Vietnamese hospital data), class consolidation, no external validation set

---


