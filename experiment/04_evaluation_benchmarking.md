# 04 — Evaluation & Benchmarking Protocol

> **Purpose.** Define every metric precisely, specify statistical tests,
> describe the benchmarking procedure, and provide a framework for
> interpreting results — including negative results.  Written to
> NeurIPS-submission standard so numbers can transfer directly into the paper.

---

## 1  Metrics Definitions

### 1.1  Primary Metrics

#### M1 — Task Success Rate (TSR)

**Definition.** Fraction of evaluation rollouts where the robot completes the
full task within the episode time limit.

\[
\text{TSR}_{\text{task}} = \frac{\text{successful episodes}}{\text{total episodes}}
\tag{1}
\]

\[
\text{TSR}_{\text{overall}} = \frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} \text{TSR}_t
\tag{2}
\]

**Protocol:**
- 20 rollout episodes per task, per condition, per seed.
- Success determined by LIBERO's built-in task-completion checker (binary).
- Report: mean ± std over 3 random seeds.
- With 3 tasks × 20 episodes × 3 seeds = 180 rollouts per condition.

**Why 20 episodes?** At 20 episodes, a true TSR of 50% has a 95% CI width of
±22 percentage points (binomial). At 50 episodes it narrows to ±14 pp.
20 is the minimum for a coarse signal; note the wide CIs in discussion.

#### M2 — Attention Overlap Score (AOS)

**Definition.** Intersection-over-Union between the top-25% attention patches
and the ground-truth object bounding box projected onto the patch grid.

Let \(S = \{i : A_i \ge A_{(k)}\}\) be the set of top-*k* saliency patches,
and \(G\) be the set of patches overlapping with the ground-truth bounding box.

\[
\text{AOS} = \frac{|S \cap G|}{|S \cup G|}
\tag{3}
\]

**Protocol:**
- Compute on 50 validation samples (evenly across tasks).
- Ground-truth bounding boxes: for LIBERO-Spatial, derive from the simulator's
  object state (position → project to image → bounding box → patch grid).
  If not available, manually annotate 50 frames (one-time cost).
- Report: mean ± std.

**Interpretation scale:**

| AOS range | Meaning |
|-----------|---------|
| < 0.10 | Random-level overlap; attention is not grounding |
| 0.10 – 0.25 | Weak grounding; some signal but mostly noise |
| 0.25 – 0.40 | Moderate grounding; attention covers part of target |
| > 0.40 | Strong grounding; attention concentrates on target |

#### M3 — Inference Latency

**Definition.** Wall-clock time for a single forward pass (image + instruction
→ action prediction), measured on the evaluation device.

**Protocol:**
- 10 warmup passes (discard).
- 100 timed passes.
- Use `torch.cuda.synchronize()` before and after.
- Report: mean, std, p95 (in milliseconds).
- Device: T4 GPU (Colab) or equivalent.

**Why this matters:** Robot manipulation typically requires ≥ 5 Hz control
frequency (200 ms/step).  If latency > 200 ms, the method is not real-time
viable on this hardware.

---

### 1.2  Secondary Metrics

#### M4 — Reconstruction MSE

\[
\text{MSE}_{\text{recon}} = \frac{1}{|M|}\sum_{i \in M}
\|\hat{\mathbf{p}}_i - \mathbf{p}_i\|^2
\tag{4}
\]

on the validation set.  Tracks whether the decoder is actually learning to
reconstruct.

**Key diagnostic:** If MSE is very low but TSR does not improve (vs baseline),
the reconstruction task is *too easy* — the decoder solved it without
improving the backbone.

#### M5 — Per-DoF Action Accuracy

\[
\text{Acc}_d = \frac{1}{N}\sum_{n=1}^{N}
\mathbb{1}\bigl[\hat{a}_{n,d} = a_{n,d}\bigr],
\qquad d = 1,\dots,7
\tag{5}
\]

Exact-bin match per DoF.  With 256 bins, random accuracy is 0.39%.
Report per-DoF and mean-across-DoF.

**Diagnostic use:** If some DoFs improve and others do not, the reconstruction
auxiliary may be helping only certain spatial dimensions (e.g., x/y position
but not gripper open/close).

#### M6 — Attention Instruction-Discrimination (AID)

For the same image \(\mathbf{x}\), compute saliency maps under two different
instructions \(\ell_1, \ell_2\) that refer to different objects.

\[
\text{AID} = 1 - \text{cosine\_similarity}(A(\ell_1), A(\ell_2))
\tag{6}
\]

**Interpretation:**
- AID ≈ 0 → Attention is identical regardless of instruction (degenerate).
- AID > 0.3 → Attention is instruction-conditioned (healthy).

**Protocol:** 20 image pairs where each image has two valid instructions
referring to different objects (e.g., "pick the bowl" vs "open the drawer"
in a scene containing both).

#### M7 — Attention Drift ΔKL

Track how much the saliency distribution changes between consecutive
evaluation checkpoints.

\[
\Delta_{\text{KL}}(t) = D_{\text{KL}}\bigl(A^{(t)} \| A^{(t-1)}\bigr)
\tag{7}
\]

Averaged over 20 samples.  Measured at epochs 1, 5, 10, 15, 20.

**Healthy:** ΔKL decreases over training (attention is stabilising).
**Unhealthy:** ΔKL oscillates or increases (loop instability).

---

## 2  Statistical Testing

### 2.1  Pairwise condition comparisons

For TSR (the primary metric), use a **paired permutation test** (two-sided)
to compare each condition pair.

**Why not a t-test?**  TSR is a proportion bounded in [0, 1].  With 20
episodes per task, the distribution may be non-normal.  Permutation tests are
non-parametric and valid for small samples.

**Procedure:**
1. For conditions A and B, collect per-task TSR differences:
   \(\delta_t = \text{TSR}^A_t - \text{TSR}^B_t\), for \(t = 1,\dots,3\) tasks.
2. Compute observed mean difference \(\bar{\delta}\).
3. Permute condition labels 10,000 times; compute null distribution.
4. p-value = fraction of permutations with |mean diff| ≥ |observed|.
5. Significance threshold: α = 0.05.  Apply **Holm-Bonferroni** correction
   for multiple comparisons.

**Number of comparisons.** With 5 conditions, there are 10 pairwise
comparisons.  After Holm-Bonferroni, the smallest p-value must be < 0.005,
the second < 0.0056, etc.

### 2.2  Effect size

Report **Cohen's h** for proportion differences:

\[
h = 2\arcsin\sqrt{\text{TSR}_A} - 2\arcsin\sqrt{\text{TSR}_B}
\tag{8}
\]

| |h| | Interpretation |
|-----|---------------|
| < 0.2 | Small effect |
| 0.2 – 0.5 | Medium effect |
| > 0.5 | Large effect |

### 2.3  Confidence intervals

For each TSR, report the **Wilson score interval** (better than Wald for
proportions near 0 or 1):

\[
\tilde{p} = \frac{x + z^2/2}{n + z^2},
\qquad
\text{CI} = \tilde{p} \pm z\sqrt{\frac{\tilde{p}(1-\tilde{p})}{n + z^2}}
\tag{9}
\]

where \(x\) = successes, \(n\) = episodes, \(z\) = 1.96 for 95% CI.

---

## 3  Benchmarking Protocol

### 3.1  Comparison table (main results)

| Condition | TSR (T1) | TSR (T2) | TSR (T3) | TSR (Overall) | AOS | Latency (ms) | Recon MSE |
|-----------|----------|----------|----------|---------------|-----|-------------|-----------|
| C1: Action-only | — ± — | — ± — | — ± — | — ± — | N/A | — ± — | N/A |
| C2: Random MAE | — ± — | — ± — | — ± — | — ± — | — ± — | — ± — | — ± — |
| C3: Naive attn MAE | — ± — | — ± — | — ± — | — ± — | — ± — | — ± — | — ± — |
| C4: Selected-head MAE | — ± — | — ± — | — ± — | — ± — | — ± — | — ± — | — ± — |
| C5: EMA teacher | — ± — | — ± — | — ± — | — ± — | — ± — | — ± — | — ± — |

All ± values are std over 3 seeds.

### 3.2  Ablation tables

**Table 2: λ ablation** (on best condition from Table 1)

| λ | TSR (Overall) | Recon MSE | Action Acc |
|---|--------------|-----------|------------|
| 0.1 | | | |
| 0.5 | | | |
| 1.0 | | | |

**Table 3: Masking ratio ablation** (on best condition)

| Mask ratio | TSR (Overall) | Recon MSE | AOS |
|------------|--------------|-----------|-----|
| 15% | | | |
| 25% | | | |
| 35% | | | |

### 3.3  Figures to produce

| # | Figure | What it shows | Key insight |
|---|--------|---------------|-------------|
| F1 | Training curves (L_total, L_action, L_recon) | All 5 conditions overlaid | Convergence comparison |
| F2 | Attention heatmap grid | 4 images × 5 conditions | Visual evidence of grounding |
| F3 | AOS over training epochs | Line plot, all conditions | Does attention sharpen? |
| F4 | AID over training epochs | Line plot, C3 vs C4 vs C5 | Is attention instruction-conditioned? |
| F5 | Inference latency bar chart | C1 vs C4 vs hypothetical diffusion | H2 claim |
| F6 | Per-task TSR grouped bar chart | T1 vs T2 vs T3, all conditions | Relational vs object-centric |
| F7 | Reconstruction quality examples | Masked image, reconstruction, original | Qualitative decoder performance |

---

## 4  Result Interpretation Framework

### 4.1  Evidence classification

Each piece of evidence is classified as:

| Classification | Definition |
|---------------|------------|
| **Strong support** | Statistically significant (p < 0.05 after correction) AND medium-to-large effect size (|h| > 0.2) |
| **Weak support** | Trend in predicted direction but not statistically significant, or small effect size |
| **Neutral** | No meaningful difference |
| **Weak evidence against** | Trend opposite to prediction but not significant |
| **Strong evidence against** | Significant effect in opposite direction |

### 4.2  Hypothesis adjudication

#### H1 — Task Success

| Evidence | Verdict |
|----------|---------|
| C4 TSR significantly > C1 TSR (p < 0.05) | Supported: reconstruction with attention masking improves action prediction |
| C4 TSR ≈ C1 TSR | Not supported at this scale; report as inconclusive |
| C4 TSR < C1 TSR | Refuted: attention masking hurts performance |

**Nuance:** The original H1 claimed "within 5% of ReconVLA." We cannot test
this directly (no ReconVLA baseline at our scale).  Reframe as: "Does
attention-masked reconstruction improve over no reconstruction?"

#### H2 — Inference Latency

| Evidence | Verdict |
|----------|---------|
| MAE forward pass < 50 ms on T4 | Supported: single-pass is fast enough for ~20 Hz control |
| MAE forward pass > 200 ms | Not viable for real-time; report as limitation |

**Note:** We compare against *theoretical* diffusion latency (T × single-pass
time) since we do not implement a diffusion baseline.

#### H3 — Attention Focus

| Evidence | Verdict |
|----------|---------|
| AOS(C4) > AOS(C1) AND AOS(C4) > AOS(C2) | Supported: attention masking concentrates attention on task-relevant objects |
| AOS(C4) ≈ AOS(C2) > AOS(C1) | Partial: reconstruction helps attention but mask source is irrelevant |
| AOS does not increase for any condition | Not supported: reconstruction does not improve grounding at this scale |

### 4.3  Critical comparisons and what they prove

| Comparison | Tests | If A wins | If B wins |
|------------|-------|-----------|-----------|
| C2 vs C1 | Does reconstruction help at all? | Any auxiliary reconstruction improves backbone representations | Reconstruction is not useful at small scale |
| C3 vs C2 | Does naive attention beat random? | Model's attention has useful signal even without head selection | Naive attention is noisy; head selection is needed |
| C4 vs C2 | **Main claim.** Does language-conditioned masking beat random? | **Core LA-ReconVLA hypothesis validated** | Language-conditioned masking is not better than random |
| C4 vs C3 | Does head selection matter? | Head selection is critical (consistent with [14]) | Head selection is unnecessary for this backbone |
| C5 vs C4 | Does mask stabilisation help? | Loop instability is a real problem; EMA is needed | Simple online masking is sufficient |
| T2,T3 vs T1 | Object-centric vs relational tasks | LA-ReconVLA better at single-object grounding | Relational grounding also improves (or neither does) |

---

## 5  Handling Negative Results

Negative results are publishable and valuable.  For each possible negative
outcome, here is the scientifically honest framing:

### 5.1  "Reconstruction does not help at all" (C2 ≈ C1)

**Framing:** At the 150-demo scale on LIBERO-Spatial, auxiliary reconstruction
does not improve action prediction.  Possible explanations:
- The 25% masking ratio is too easy (Sec. 3.3 of 01_theoretical_analysis).
- 20 epochs is insufficient for the reconstruction signal to propagate to
  action-relevant representations.
- The dataset is too small for the auxiliary task to generalise.

**Next step:** Report and recommend testing at higher masking ratios or on
a larger data subset.

### 5.2  "Attention masking is no better than random" (C4 ≈ C2)

**Framing:** The reconstruction auxiliary itself is beneficial, but the
model's cross-attention does not provide a *better* masking signal than
random at this scale.  This is consistent with:
- SemMIM's finding that text-guided masking needs deep text involvement [19].
- The small data regime may not allow the attention loop to develop.

**What this still contributes:** Confirmation that reconstruction helps;
a well-documented negative on attention-derived masking with analysis of
head entropy and AOS evolution.

### 5.3  "Naive attention hurts, but head selection rescues" (C3 < C2, C4 > C3)

**Framing:** This is actually a *positive and publishable* finding:
"Attention-based mask selection for VLA reconstruction works, but *only*
with careful head selection.  Naive averaging across all heads is
counterproductive due to attention sinks and diffuse heads."

**Contribution:** Methodological insight connecting localization-head
literature [14] to VLA reconstruction.

### 5.4  "Nothing helps" (C1 ≈ C2 ≈ C3 ≈ C4 ≈ C5)

**Framing:** Auxiliary reconstruction and attention-guided masking do not
improve VLA performance on LIBERO-Spatial at the 150-demo scale.  Analyse
*why*:
- Was reconstruction MSE already near zero? (Too easy.)
- Were attention maps unchanged? (Backbone too frozen.)
- Were all tasks already near-ceiling or near-floor? (Dataset saturation.)

**Contribution:** Negative result with thorough ablation and failure-mode
analysis.  Identifies bottleneck (scale, backbone, task difficulty) for
future work.

---

## 6  Reproducibility Checklist (NeurIPS Standard)

| Item | Status | Notes |
|------|--------|-------|
| Random seeds specified | Yes | 42, 123, 7 |
| Hyperparameters documented | Yes | train_config.yaml + Appendix B of Part 1 |
| Dataset publicly available | Yes | LIBERO via HuggingFace |
| Code provided | Yes | GitHub repo (zipped with submission) |
| Hardware specified | Yes | Colab T4 (15 GB) or Kaggle P100 |
| Training time reported | Per condition | ~90 min on T4 |
| Evaluation episodes specified | Yes | 20 per task per seed |
| Statistical tests specified | Yes | Paired permutation, Holm-Bonferroni |
| Confidence intervals reported | Yes | Wilson score intervals |
| Negative results documented | Yes | Section 5 of this document |

---

## 7  Final Report Mapping

This section maps each metric and experiment to the paper sections where
results will appear.

| Paper section | Content source |
|--------------|----------------|
| §4.1 Main Results | Table 1 (TSR), Figure F1 (curves), Figure F6 (per-task) |
| §4.2 Attention Analysis | AOS table, Figures F2 (heatmaps), F3 (AOS over time), F4 (AID) |
| §4.3 Efficiency | Latency table, Figure F5 |
| §4.4 Ablation Studies | Tables 2–3, per-condition discussion |
| §4.5 Reconstruction Quality | Recon MSE table, Figure F7 |
| §5 Discussion | Hypothesis verdicts (Sec. 4.2 above), negative result analysis (Sec. 5) |

---

## References

- [1] Song et al., "ReconVLA," arXiv:2508.10333, 2025.
- [4] He et al., "Masked autoencoders," CVPR, 2022.
- [14] Zhang et al., "Localization heads," arXiv:2503.06287, 2025.
- [19] Zeng et al., "SemMIM" (cited in feedback analysis).
