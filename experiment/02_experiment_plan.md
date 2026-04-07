# 02 — Experiment Plan: Step-by-Step Protocol

> **Purpose.** A complete, reproducible experiment protocol.  For every
> condition: what we run, what we observe, and what each possible outcome
> *means* (supports, refutes, or is ambiguous for the LA-ReconVLA hypothesis).
> Written so that a second researcher could replicate the study from this
> document alone.

---

## 1  Experimental Question (Refined)

The Part 1 report asked:
> "Can gaze-region supervision be replaced by language-driven attention
> masking without losing task performance?"

Following the expert feedback [feedback.md], we **narrow** this to a cleaner,
budget-feasible question:

> **Under the same small compute budget, does language-conditioned masking help
> more than generic reconstruction?**

This removes the comparison with full-scale ReconVLA (8× A100) and instead
produces interpretable ablation results within a single Colab T4 session.

---

## 2  Experimental Conditions

Five conditions, each differing in exactly one dimension from its neighbours.
The progression is designed to isolate three effects: (A) reconstruction
itself, (B) semantic mask selection, and (C) head selection.

| ID | Condition | Mask source | Decoder | λ | What it isolates |
|----|-----------|-------------|---------|---|------------------|
| **C1** | Action-only baseline | None | None | 0 | Lower bound: VLA without any reconstruction auxiliary |
| **C2** | Random-mask MAE | Random 25% patches | 4-layer MAE | 0.5 | Effect (A): does *any* reconstruction help? |
| **C3** | Naive attention-mask MAE | All-head averaged cross-attn, top-25% | 4-layer MAE | 0.5 | Effect (B): does attention-guided masking beat random? |
| **C4** | Selected-head attention-mask MAE | Localization-head cross-attn, top-25% | 4-layer MAE | 0.5 | Effect (C): does head selection matter? |
| **C5** | Selected-head + EMA teacher | Same as C4 but masks from EMA backbone | 4-layer MAE | 0.5 | Stability: does mask stabilisation help? |

**Ablation extensions** (run after the five main conditions if time permits):

| ID | Ablation | Variable | Values |
|----|----------|----------|--------|
| **A1** | λ sweep on best condition | λ | 0.1, 0.5, 1.0 |
| **A2** | Masking ratio sweep | *k* / *P* | 15%, 25%, 35% |
| **A3** | Contiguous vs scattered masks | Mask topology | Scattered top-*k* vs connected-component region |

---

## 3  Fixed Experimental Parameters

All conditions share:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Backbone | PaliGemma-3B (frozen, last 2 layers fine-tuned) | Part 1 spec; fits T4 in fp16 |
| Image size | 224 × 224 | ViT-standard |
| Patch grid | 14 × 14 = 196 | 16 px patches |
| Action discretisation | 256 bins × 7 DoF | OpenVLA convention [2] |
| Batch size | 8, grad accum ×4 (eff. 32) | T4 VRAM constraint |
| Optimiser | AdamW, lr = 1e-4, cosine schedule, 100 warmup steps | Standard VLM fine-tuning |
| Epochs | 20 | ~90 min on T4 per condition |
| Dataset | LIBERO-Spatial, 3 tasks × 50 demos | Part 1 spec |
| Train/val split | 85% / 15% per task | Part 1 spec |
| Random seeds | 42, 123, 7 (report mean ± std over 3 seeds) | Statistical robustness |

**Tasks selected:**

| # | Task ID | Type | Why chosen |
|---|---------|------|------------|
| T1 | `KITCHEN_SCENE1_put_the_black_bowl_in_the_top_drawer_of_the_cabinet` | Place (relational) | Two-region grounding: bowl + drawer |
| T2 | `KITCHEN_SCENE2_open_the_bottom_drawer_of_the_cabinet` | Single-object | Single-region grounding: drawer |
| T3 | `KITCHEN_SCENE3_turn_on_the_stove` | Single-object | Single-region grounding: stove knob |

This mix allows us to test prediction P3 (relational vs object-centric).

---

## 4  Step-by-Step Protocol

### Step 0 — Environment & Data (Day 0)

**Actions:**
1. Set up Colab notebook; install dependencies; mount Drive.
2. Download LIBERO-Spatial 3-task subset (~900 MB).
3. Verify data loading: print sample counts, visualise 5 random frames per
   task, confirm image shapes and action ranges.
4. Load PaliGemma-3B tokenizer + model in fp16; verify it fits in T4 VRAM.
5. Run a single forward pass with a dummy batch; confirm output shapes.

**Checkpoint:** Move to Step 1 only when forward pass succeeds and data loads
correctly.  Record VRAM usage.

---

### Step 1 — Attention Map Extraction Diagnostic (Day 1, ~30 min)

**Goal:** Before training anything, verify that PaliGemma's cross-attention
contains *any* usable spatial signal.

**Actions:**
1. For 20 random (image, instruction) pairs across all 3 tasks:
   - Extract cross-attention maps from the *frozen* backbone (all layers, all heads).
   - For each head *h* in each layer *l*, compute spatial entropy:
     \(H_h = -\sum_i \alpha_i \log \alpha_i\)
   - Identify candidate **localization heads**: heads with high mean attention
     magnitude on image tokens AND low spatial entropy [14].
2. Visualise the top-3 localization heads as heatmaps overlaid on the image.
3. Compute a preliminary AOS (attention overlap score) for these heads against
   the rough object bounding box (manual annotation for 20 images is fine at
   this stage — or use the task semantics to heuristically identify the
   approximate object region).

**Observables:**

| Observable | Favours LA-ReconVLA | Against LA-ReconVLA |
|------------|--------------------|--------------------|
| ≥ 2 heads with spatial entropy < median AND high image-attn magnitude | Yes: localizing heads exist | — |
| AOS of best heads > 0.15 on frozen backbone | Yes: pseudo-gaze has signal before any training | — |
| No head shows spatial selectivity (all uniform) | — | Severe: backbone lacks grounding signal; attention-masking will be random-like |
| High-attn patches are all sinks (corners, edges) | — | Moderate: need sink filtering; head selection is essential |

**Decision gate:**
- If **no localizing heads found** → the backbone is too weakly grounded.
  Fall back to random masking + action loss (C2 becomes the best we can do).
  Document this as a *negative* but informative result.
- If **localizing heads found** → proceed with the full protocol.

---

### Step 2 — Condition C1: Action-Only Baseline (Day 1, ~90 min)

**Actions:**
1. Train the baseline VLA (PaliGemma backbone + action head, no MAE decoder,
   no masking) for 20 epochs.
2. Log: train loss, val loss, action accuracy per DoF, per epoch.
3. Save best checkpoint.

**Observables:**

| Metric | Expected range | Notes |
|--------|---------------|-------|
| Val action loss convergence | Decreasing over 20 epochs | If not: lr or data issue |
| Per-DoF accuracy | 15–40% (modest, small data) | Establishes lower bound |

---

### Step 3 — Condition C2: Random-Mask MAE (Day 2, ~90 min)

**Actions:**
1. Add MAE decoder to the model. Mask = random 25% patches (uniform, no
   attention involvement).
2. Train for 20 epochs with λ = 0.5.
3. Log: train total/action/recon loss, val losses.

**Observables and interpretation:**

| Outcome | Interpretation |
|---------|---------------|
| C2 action loss < C1 action loss | **Reconstruction helps.** Generic MAE auxiliary improves action prediction — validates the reconstruction-as-regulariser principle. |
| C2 ≈ C1 | Reconstruction alone is not enough at this scale. Two possibilities: (a) 25% masking is too easy (theory Sec. 3.3); (b) 20 epochs insufficient. |
| C2 > C1 (worse) | Reconstruction loss is *hurting* — λ too high, or decoder is stealing gradient from action head. Try λ = 0.1. |
| Recon loss decreases but action loss does not | Decoder solves reconstruction without improving backbone — the "too easy" failure mode. |

---

### Step 4 — Condition C3: Naive Attention-Mask MAE (Day 2, ~90 min)

**Actions:**
1. Replace random masking with naive attention masking: average cross-attention
   across *all* heads and last 3 layers, take top-25%.
2. Train for 20 epochs with λ = 0.5.
3. Log all losses + saliency map snapshots every 5 epochs.

**Observables and interpretation:**

| Outcome | Interpretation |
|---------|---------------|
| C3 > C2 (better action accuracy) | **Core hypothesis supported:** attention-guided masking is better than random masking. The model's internal attention provides useful supervision. |
| C3 ≈ C2 | Attention masking adds no value over random — consistent with naive averaging mixing signal with noise (predicted by theory, Sec. 2.2 F1). Head selection is needed. |
| C3 < C2 (worse) | Naive attention masks are *harmful* — likely dominated by sinks or degenerate loop. Confirms the "naive averaging is dangerous" warning [14, 15]. |
| Saliency maps sharpen over training | Healthy loop: backbone is learning to attend to task-relevant regions. Virtuous cycle confirmed. |
| Saliency maps collapse to fixed pattern regardless of instruction | **Degenerate loop.** Backbone ignores instruction. Confirms F3 (frozen backbone bias). |

---

### Step 5 — Condition C4: Selected-Head Attention-Mask MAE (Day 3, ~90 min)

**Actions:**
1. Use the localization heads identified in Step 1. Aggregate attention over
   *only* those heads (typically 2–5 heads across last 3 layers).
2. Otherwise identical to C3.

**Observables and interpretation:**

| Outcome | Interpretation |
|---------|---------------|
| C4 > C3 | **Head selection is critical.** This is the single most important finding — confirms localization-heads theory [14] and validates the refined masking recipe. |
| C4 ≈ C3 | Head selection does not matter much — either (a) PaliGemma's heads are relatively uniform, or (b) the learnable head weights in AttentionGuidedMasker already down-weight bad heads. |
| C4 < C3 | Unexpected. Selected heads may not be the right ones — revisit selection criteria. |

**Critical comparison: C4 vs C2.**

| Outcome | Interpretation |
|---------|---------------|
| C4 > C2 | **Main result:** Language-conditioned masking (with head selection) provides meaningful supervision beyond generic reconstruction. The core LA-ReconVLA idea is validated. |
| C4 ≈ C2 | Reconstruction itself is the main driver; mask source is secondary. The idea is *not refuted* but is weaker than hoped. |
| C4 < C2 | Attention-derived masks are actively harmful compared to random. Indicates backbone's attention is too unreliable for self-supervision at this scale. |

---

### Step 6 — Condition C5: EMA Teacher Masks (Day 3, ~90 min)

**Actions:**
1. Maintain an EMA copy of the backbone (β = 0.999).
2. Compute attention masks from the EMA backbone (detached, no gradients).
3. Train the main backbone with these stabilised masks.

**Observables and interpretation:**

| Outcome | Interpretation |
|---------|---------------|
| C5 > C4 | **Mask instability was the bottleneck.** EMA stabilisation is necessary. Confirms the self-confirming loop risk (theory Sec. 4). |
| C5 ≈ C4 | Loop was already stable enough — the learnable head weights or the small data scale prevented oscillation. |
| C5 < C4 | EMA slows adaptation too much. The backbone needs faster mask updates at this training scale. Try lower β (0.99). |

---

### Step 7 — Per-Task Analysis (Day 4, ~1 hr)

**Actions:**
1. For the best condition (C4 or C5), compute per-task metrics.
2. Compare T1 (place task) vs T2, T3 (single-object tasks).

**Observables and interpretation:**

| Outcome | Interpretation |
|---------|---------------|
| Improvement on T2, T3 but not T1 | **Predicted by theory (P3):** Attention-guided masking helps object-centric tasks but misses the destination in relational tasks. RoboGround [16] explains this. |
| Improvement on all tasks | Stronger-than-expected result: cross-attention captures both object and destination. |
| Improvement on T1 only | Surprising. The relational task may benefit most from spatial regularisation. Investigate which region (bowl vs drawer) the mask captures. |

---

### Step 8 — Ablations (Day 4–5, time permitting)

**A1: λ sweep** on best condition.

| λ | Expected effect |
|---|----------------|
| 0.1 | Reconstruction barely influences backbone. Action loss dominates. |
| 0.5 | Balanced. Default. |
| 1.0 | Reconstruction dominates. Risk: action accuracy degrades as backbone optimises for reconstruction. |

**A2: Masking ratio sweep.**

| Ratio | Expected effect |
|-------|----------------|
| 15% | Too easy; decoder interpolates; weak gradient signal (Eq. 9). |
| 25% | Default. |
| 35% | Harder task; stronger gradient signal; but more information removed — may hurt action accuracy if backbone cannot compensate. |

**A3: Contiguous vs scattered masks** (if time).
Convert top-*k* patches to a connected component via dilation on the 14×14
grid.  Compare AOS and action loss.

---

## 5  Metrics Collected for Every Condition

| Metric | Logged | Frequency | Purpose |
|--------|--------|-----------|---------|
| **Train total loss** | WandB | Every step | Convergence monitoring |
| **Train action loss** | WandB | Every step | Action learning signal |
| **Train recon loss** | WandB | Every step | Reconstruction learning signal |
| **Val total loss** | WandB | Every epoch | Generalisation |
| **Val action accuracy (per-DoF)** | WandB | Every epoch | Per-dimension prediction quality |
| **Saliency map snapshot** | PNG to Drive | Every 5 epochs | Visual inspection of attention evolution |
| **Mask instruction-discrimination** | Table | Epoch 1, 10, 20 | Do different instructions produce different masks on the same image? |
| **Attention Overlap Score (AOS)** | Table | End of training | Quantitative grounding quality |
| **Inference latency (ms)** | Table | End of training | H2: efficiency claim |
| **VRAM usage** | Manual note | Start of each condition | Reproducibility |
| **Wall-clock training time** | Manual note | End of each condition | Practical cost |

---

## 6  Master Decision Tree

```
START
  │
  ├── Step 1: Are there localizing heads?
  │     ├── NO  → Report negative result (backbone lacks grounding signal).
  │     │         Run C1 + C2 only. Conclude: attention masking infeasible
  │     │         for this backbone at this scale.
  │     └── YES → Continue.
  │
  ├── Step 3: Does C2 beat C1?
  │     ├── NO  → Reconstruction not helpful at this scale/λ.
  │     │         Try λ = 0.1 or higher masking ratio.
  │     │         If still no: auxiliary reconstruction does not help on 
  │     │         LIBERO-Spatial at 3×50 demo scale. Report as finding.
  │     └── YES → Continue.
  │
  ├── Step 5: Does C4 beat C2?
  │     ├── YES → **Core hypothesis validated.**
  │     │         Language-conditioned masking > random masking.
  │     │         Report as main result.
  │     ├── ≈   → Reconstruction helps, but mask source is secondary.
  │     │         Moderate result. Report honestly.
  │     └── NO  → Attention masking harms. Report as negative.
  │             Analyse *why* (sinks? loop collapse? relational tasks?).
  │
  ├── Step 6: Does C5 beat C4?
  │     ├── YES → Stabilisation needed. EMA teacher is recommended.
  │     └── NO  → C4 is sufficient. Simpler method preferred.
  │
  └── Step 7: Per-task breakdown
        ├── Object tasks improved, relational not → Expected (P3).
        ├── All improved → Strong positive.
        └── None improved → Method needs fundamental redesign.
```

---

## 7  Timeline

| Day | Activity | GPU hours | Output |
|-----|----------|-----------|--------|
| 0 | Setup, data, forward-pass smoke test | 0.5 | Verified environment |
| 1 | Step 1 (attention diagnostic) + Step 2 (C1 baseline) | 2 | Localization heads identified; baseline metrics |
| 2 | Step 3 (C2) + Step 4 (C3) | 3 | Random-mask and naive attention-mask results |
| 3 | Step 5 (C4) + Step 6 (C5) | 3 | Selected-head and EMA results |
| 4 | Step 7 (per-task) + Step 8 (ablations A1, A2) | 2 | Full ablation table |
| 5 | Visualisations, result compilation, sanity checks | 0.5 | Figures, tables, metrics JSON |
| **Total** | | **~11 hrs** | Fits in 2 free Colab sessions |

---

## 8  What Each Final Outcome Means for the Paper

| Scenario | Paper narrative | NeurIPS-style contribution |
|----------|----------------|---------------------------|
| C4 > C2 > C1, clear margin | "Annotation-free language-attention masking provides meaningful grounding supervision for VLAs beyond generic reconstruction." | Positive result: annotation-free alternative to gaze-based grounding. |
| C4 ≈ C2 > C1 | "Reconstruction auxiliary improves VLA performance, but language-conditioned mask selection does not significantly outperform random masking at this scale." | Nuanced result: reconstruction helps, mask source is secondary. Opens question of whether head selection matters at scale. |
| C2 > C1, C3 < C2, C4 > C3 | "Naive attention aggregation is harmful, but careful head selection recovers a useful signal." | Methodological insight: *how* to use attention matters more than *whether* to use it. |
| C1 ≈ C2 ≈ C3 ≈ C4 | "Reconstruction auxiliary and attention masking do not improve VLA performance at the 150-demo scale on LIBERO-Spatial." | Negative result: publishable if well-analysed. Identify scale, masking difficulty, or backbone grounding as bottleneck. |

All scenarios are publishable if honestly analysed with proper ablations.

---

## References

- [1] Song et al., "ReconVLA," arXiv:2508.10333, 2025.
- [2] Kim et al., "OpenVLA," arXiv:2406.09246, 2024.
- [4] He et al., "Masked autoencoders are scalable vision learners," CVPR, 2022.
- [14] Zhang et al., "Localization heads," arXiv:2503.06287, 2025.
- [15] Chen et al., "Visual attention sink," arXiv:2503.03321, 2025.
- [16] Huang et al., "RoboGround," CVPR, 2025.
- [17] Counterfactual failures in VLAs, arXiv:2602.17659, 2026.
