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

## Paper Remaining ✍️
1. Section V — Experiments & Results
2. Implementation details (GPU, versions, GitHub)
3. Baseline comparison table — each backbone vs stack
4. Ablation study — 3 ablations proving each component contributes
5. Threshold analysis — MCC vs threshold curve
6. Feature importance — which model and metadata contribute most
7. Per-pathology breakdown on test set

## References - Four marked [REPLACE] — these need your literature search:
[2] — inter-reader variability paper
[4] — radiologist fatigue paper
[22] — multi-resolution stacking preprint
[29] — the PMC6476887 metadata signal paper (find the exact title)

## Next steps in order:

1. Run the training pipeline
2. Fill in all [—] result placeholders
3. Replace the four [REPLACE] references
4. Replace all [CITE X] tags in the body with their [number] equivalents
5. Post to arXiv — submit to IEEE JBHI

### What the paper needs that you don't have yet

1. **Literature review** — you need to cite 20–30 existing chest X-ray AI papers and explain what your work adds
2. **Ablation study** — show each model's individual contribution to the ensemble (this is actually already easy with your pipeline)
3. **Statistical significance** — confidence intervals on your MCC, not just a point estimate
4. **Radiologist validation** — ideally one radiologist reviews a sample of predictions. This is where your Kaiser access is a huge advantage
5. **Limitations section** — distribution shift (Vietnamese hospital data), class consolidation, no external validation set

---
### Novelty Claim

**⭐ CORE NOVELTY STATEMENT — use this exact framing throughout the paper:**

> *"We propose a stacking ensemble framework that combines heterogeneous vision model predictions with patient metadata via a learned meta-learner, and demonstrate that this architecture outperforms single-model baselines on binary chest X-ray triage — a clinically critical screening task"*

This framing is what keeps the paper novel. The **task** (binary chest X-ray classification) is known and published. The **architecture** is the contribution. Never lead with the task — always lead with the architecture.

How to say it to a skeptical reviewer:
If a reviewer says "stacking has been done before" — your response is:

*"We agree stacking is an established technique. Our contribution is the first application of architecturally diverse stacking with patient-level OOF leakage prevention and demographic metadata integration to binary chest X-ray triage on the NIH ChestX-ray14 benchmark. Each of these three design choices is individually motivated, and our ablation study (Table V) demonstrates that each contributes measurably to overall MCC."*

That response is airtight because it is specific, it points to your ablation table as evidence, and it does not overclaim.

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
**Four Specific Sub-Claims**
1. Sub-claim 1 — Architectural diversity in the ensemble
Prior stacking in chest X-ray uses variants of the same backbone family (e.g., EfficientNet-B0/B1/B2). We use six architecturally diverse backbones spanning three design families — convolutional, hybrid, and pure transformer. This diversity produces uncorrelated errors that the meta-learner exploits. No prior published work on chest X-ray binary triage has done this specific combination.
2. Sub-claim 2 — Leakage-free OOF at patient level
Prior stacking work trains the meta-learner on in-sample predictions — causing data leakage. We prevent this with patient-level stratified OOF cross-validation. The patient-level stratification additionally prevents follow-up scan leakage — a known but frequently ignored problem in ChestX-ray14 benchmarking.
3. Sub-claim 3 — Demographic metadata as a tabular modality
Age and sex are available in every radiology record but consistently discarded in ensemble pipelines. We integrate them as tabular features in CatBoost — not injected into the neural network. Prior work that includes metadata concatenates it inside the neural network, adding architectural complexity and instability.
4. Sub-claim 4 — MCC-optimised threshold selection
Almost all chest X-ray papers use AUC-ROC and a default 0.5 threshold. We use MCC as the early stopping criterion during training and grid-search for the MCC-optimal threshold. This is a methodological contribution to how chest X-ray triage models should be evaluated.

What We Are NOT Claiming

1. Not claiming to invent stacking (Wolpert 1992)
2. Not claiming to invent OOF (standard ML practice)
3. Not claiming to invent any of the six backbones
4. Not claiming the binary triage task is new
5. Not claiming state-of-the-art on all 14 disease labels

---

**Related Work structure that makes novelty airtight (3 paragraphs):**

- **Para 1** — Single backbone approaches (CheXNet, DenseNet, ViT). Limitation: single model, no ensemble diversity.
- **Para 2** — Ensemble methods in medical imaging. Show existing ensembles use simple averaging/voting — not learned stacking. Limitation: no meta-learner, no leakage prevention, no metadata.
- **Para 3** — The gap: no prior work combines diverse vision backbone stacking + OOF + tabular metadata in a learned meta-learner for chest X-ray binary triage. Your paper fills this gap.

---



