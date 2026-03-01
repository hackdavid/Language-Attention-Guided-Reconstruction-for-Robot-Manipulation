# PART 1 — Writing Guide: Critical Appraisal & Proposal
## ReconVLA: Reconstructive Vision-Language-Action Model

> **Strict Requirements:** ~2,000 words total | PDF via Turnitin | Due: 6/3/2026 16:00  
> **Marking Weight:** 40% of module | Sections: Summary (200w) + Appraisal (800w) + Proposal (1000w)

---

## How to Use This Guide

This document tells you **what to write, in what order, and why** for each section. Follow the word budgets. Do not exceed 2,200 words total. Every claim must cite from the VLA reading sequence or peer-reviewed sources. Use **Harvard referencing** throughout.

---

## SECTION 1 — Summary (~200 words)

**Goal:** Demonstrate you understand what ReconVLA *does* and *why it matters*, not just what it claims.

**Write in this sequence:**

1. **State the problem** ReconVLA solves (1–2 sentences):  
   Existing VLA models produce *dispersed visual attention* — they fail to focus on the task-relevant manipulation target. This is the root cause of imprecise robot manipulation.

2. **State the core architectural contribution** (2–3 sentences):  
   ReconVLA introduces an *implicit grounding paradigm*. A diffusion transformer, conditioned on the VLM's visual output tokens (*reconstructive tokens*), is trained to reconstruct the gaze region of the input image — the spatial region corresponding to the target object. This auxiliary reconstruction task forces the backbone to learn fine-grained, geometrically precise representations.

3. **State the key results** (1–2 sentences):  
   Evaluated on LIBERO, CALVIN, and real-robot tasks. Outperforms OpenVLA and RT-2 style baselines in long-horizon manipulation, with improved visual attention focus visualised via attention maps.

4. **One-sentence positioning** in the broader field:  
   ReconVLA distinguishes itself from *explicit grounding* (external grounding experts) and *CoT grounding* (bounding box output) by using visual reconstruction as a purely internal, implicit supervisory signal.

**Cite here:** Song et al. (2025) arXiv:2508.10333.

---

## SECTION 2 — Critical Appraisal (~800 words)

**Goal:** Show *examiner-level insight*. Move beyond summarising — interrogate methodology, datasets, baselines, and theoretical foundations. Organise around 4 clear sub-arguments.

### 2.1 Theoretical Foundations (link to module content) — ~150 words

Connect ReconVLA to deep learning principles covered in your module:

- The reconstruction objective functions as a **self-supervised auxiliary loss** — this is analogous to the MAE framework (He et al., 2022) where masking forces representation richness.
- The diffusion transformer head leverages **DDPM-style denoising** (Ho et al., 2020) to model the conditional distribution `p(scene_tokens | reconstructive_tokens)`.
- This relates to the **representation bottleneck** concept: by forcing the VLM backbone's outputs to carry sufficient information to reconstruct a masked region, the model is regularised against learning shortcut features.
- Connect to **backpropagation**: gradients from the reconstruction loss flow back through the reconstructive tokens into the VLM backbone, reshaping its internal representations.

**Cite:** Ho et al. (2020) NeurIPS; He et al. (2022) CVPR; Radford et al. (2021) CLIP paper.

### 2.2 Methodology Critique — Gaze Region Dependency (~200 words)

This is your **strongest critique** — the one that will earn distinction marks.

**Argue the following:**

- The gaze region used as reconstruction target is defined by *robot eye-tracking or gaze annotation* from the training dataset. The paper does not clearly specify how gaze regions are obtained across the three training datasets (BridgeData V2, LIBERO, CALVIN). This is a methodological gap.
- If gaze regions are derived heuristically (e.g., bounding box around the stated object), this introduces a *circular dependency*: the reconstruction target is computed from the same language instruction used to guide the action, potentially allowing the model to shortcut by attending to language rather than developing genuine geometric understanding.
- The paper's comparison is between implicit grounding (ReconVLA) vs explicit grounding and CoT grounding — but **no ablation isolating the effect of the gaze region definition** is provided. Does reconstruction of a *random region* perform similarly? This is a missing ablation.

**Link to module content:** This is a **baseline fairness issue** — without ablation on region selection, we cannot attribute the performance gain to geometric understanding specifically.

### 2.3 Computational and Scalability Limitations (~200 words)

- Training requires **8 × A100 (80GB) GPUs** and 2 million samples. This is a significant barrier to academic reproduction and fine-tuning in low-resource settings.
- The **diffusion transformer** adds inference overhead. Diffusion models require iterative denoising steps (`T` forward passes per generation). In robot manipulation, where latency directly affects control frequency, this is a practical bottleneck. The paper does not report **inference latency** benchmarks — a critical omission for a robotics paper.
- **Multi-view image inputs** are required (the GitHub shows multi-view setup). Deploying in environments without calibrated multi-camera setups limits generalisability.
- Compare to: **Diffusion Policy** (Chi et al., 2023) which explicitly benchmarks inference latency — ReconVLA does not.

**Cite:** Chi et al. (2023) arXiv:2303.04137.

### 2.4 Dataset and Evaluation Scope (~250 words)

- **LIBERO** and **CALVIN** are simulation benchmarks. Real-world results are shown but limited in scope (the paper shows qualitative results on a single robot arm setup). The real-world generalisation claim is therefore not strongly evidenced.
- CALVIN evaluates long-horizon task chains (4 subtasks). While ReconVLA shows improvement here, the benchmark's **language instruction vocabulary is fixed** — it does not evaluate open-vocabulary instruction following, which is the core promise of VLA models. Connect to: **OpenVLA** (Kim et al., 2024) which addresses open-vocabulary instructions on a broader distribution.
- The **100k trajectory pretraining dataset** is assembled from BridgeData V2, LIBERO, and CALVIN — datasets that partially overlap with evaluation environments. This risks *data leakage*: the model may have seen similar visual scenes during pretraining that appear in the test distribution, inflating generalisation metrics.
- The paper **does not report FID or perceptual quality scores** for the reconstruction outputs — only task success rates. It is impossible to verify whether high task success correlates with high reconstruction quality, or whether the model finds a shortcut.

**Cite:** Kim et al. (2024) arXiv:2406.09246 (OpenVLA); Mees et al. (2022) arXiv:2112.03227 (CALVIN).

---

## SECTION 3 — Proposal for Improvement (~1,000 words)

**Goal:** Propose a technically specific, feasible improvement that directly addresses your critique. This **must** be the basis of your Part 2 implementation.

### Proposed Improvement: Language-Attention Guided Masked Reconstruction (LA-ReconVLA)

**Name your proposal clearly at the start.**

#### 3.1 Problem Identification and Motivation (~150 words)

Summarise the gap your proposal addresses. Write it as a research question:

> *"ReconVLA's visual grounding depends on heuristically defined gaze regions, which may not generalise across diverse manipulation scenarios and introduces a potential circular dependency with language conditioning. Can we replace gaze-region supervision with language-driven attention masking to derive reconstruction targets that are semantically grounded in the task instruction, while simultaneously replacing the heavyweight diffusion transformer with a computationally efficient masked autoencoder (MAE) decoder?"*

State explicitly: your proposal addresses the **gaze annotation dependency** and the **inference overhead** simultaneously.

#### 3.2 Technical Description (~400 words)

Describe the architecture precisely:

**Step 1 — Extract Cross-Attention Maps:**  
In the VLM backbone (LLaVA-style transformer), the cross-attention layers compute attention scores between language tokens and image patch tokens. At inference time, these scores indicate *which image patches the model attends to when processing the language instruction*. Aggregate attention scores across all language tokens and all layers to produce an attention saliency map `A ∈ R^{H×W}` over image patches.

**Step 2 — Attention-Guided Masking:**  
Apply a top-k threshold to `A` to select the most attended image patches. These are the patches most semantically relevant to the language instruction — analogous to gaze regions but derived *endogenously* from the model rather than exogenous annotation. Mask these patches from the visual encoder input (set patch tokens to a learnable mask token).

**Step 3 — Lightweight MAE Decoder:**  
Replace the diffusion transformer with a **shallow 4-layer transformer decoder** (MAE-style, He et al., 2022). The decoder receives: (a) unmasked patch tokens from the visual encoder, and (b) learnable mask tokens at masked positions. It reconstructs the pixel values at masked positions using an MSE reconstruction loss:

```
L_recon = MSE(decoder(unmasked_tokens, mask_tokens), original_masked_pixels)
```

**Step 4 — Joint Training:**  
The total loss is:

```
L_total = L_action + λ · L_recon
```

where `L_action` is the original cross-entropy action prediction loss and `λ` is a weighting hyperparameter (to be tuned: default `λ = 0.5`).

**Why a MAE decoder instead of diffusion?**  
The MAE decoder requires a single forward pass (no iterative denoising), reducing inference time. For geometric understanding, the reconstruction objective only needs to be *sufficient* — not photorealistic. Coarse reconstruction at correct locations is adequate to force the backbone to encode spatial structure.

**Why attention-based masking instead of gaze regions?**  
Language attention maps are derived directly from the task instruction, making the masking target semantically grounded without requiring external gaze annotations. This also generalises to novel instructions unseen during pretraining.

#### 3.3 Theoretical Justification (~200 words)

Connect to module theory:

- **Self-supervised learning theory:** MAE has been shown (He et al., 2022) to produce stronger visual representations than contrastive methods (e.g., CLIP) when reconstruction targets are semantically meaningful. By masking high-attention patches, we force the backbone to predict task-relevant content, not arbitrary textures.
- **Information bottleneck principle:** Masking high-attention patches and requiring their reconstruction creates an information bottleneck — the model must retain spatial information in its latent representation that would otherwise be dropped.
- **Gradient flow:** Unlike diffusion, where gradients from the denoising objective must traverse many timesteps before reaching the backbone, the MAE decoder provides *direct gradient signals* to the encoder — improving training stability.
- **Attention regularisation:** Using attention maps as masking targets creates an implicit regularisation loop: the attention map determines what is masked; the reconstruction loss improves the quality of the backbone features; better features produce sharper attention maps. This self-reinforcing loop encourages convergence to task-relevant visual grounding.

#### 3.4 Hypothesised Outcomes (~150 words)

State measurable hypotheses:

- **H1 (Action accuracy):** LA-ReconVLA achieves within 5% of ReconVLA's task success rate on LIBERO-Spatial, despite not using gaze annotations.
- **H2 (Inference efficiency):** LA-ReconVLA runs at 3–5× lower inference latency than ReconVLA due to the single-pass MAE decoder.
- **H3 (Attention focus):** Attention maps from LA-ReconVLA show higher concentration on task-relevant objects (measured by overlap with object bounding boxes) compared to baseline VLA without reconstruction.
- **H4 (Annotation-free generalisation):** LA-ReconVLA maintains task success on instructions outside the training distribution where gaze annotations are unavailable.

State that Part 2 will test H1, H2, and H3 using the LIBERO-Spatial benchmark with a lightweight implementation on constrained compute (Colab/Kaggle T4 GPU).

---

## Referencing Checklist (Harvard Format)

Ensure the following are cited inline and in your reference list:

| Paper | Why Cite |
|---|---|
| Song et al. (2025) arXiv:2508.10333 | ReconVLA — primary paper |
| Radford et al. (2021) ICML — CLIP | Foundation of vision-language alignment |
| Kim et al. (2023) — LLaVA arXiv:2304.08485 | VLM backbone architecture |
| Kim et al. (2024) arXiv:2406.09246 — OpenVLA | Baseline VLA, open-source comparison |
| Ho et al. (2020) NeurIPS — DDPM | Diffusion model used in ReconVLA |
| He et al. (2022) CVPR — MAE | Justifies your proposed decoder |
| Chi et al. (2023) arXiv:2303.04137 — Diffusion Policy | Inference latency comparison |
| Mees et al. (2022) arXiv:2112.03227 — CALVIN | Benchmark used in evaluation |
| Chen et al. (2021) arXiv:2106.01345 — Decision Transformer | Historical sequence in VLA development |
| Brohan et al. (2023) arXiv:2307.15818 — RT-2 | VLA backbone comparison |

---

## Quality Check Before Submission

- [ ] Summary is ≤ 220 words
- [ ] Appraisal is 750–900 words with 4 distinct sub-arguments
- [ ] Proposal names a specific architecture change (not "add more layers")
- [ ] Proposal includes a loss function or training objective formula
- [ ] Hypotheses are measurable (not vague claims)
- [ ] All claims cite a specific source with year
- [ ] Harvard referencing format used throughout
- [ ] Word count is 1,800–2,200 total
- [ ] PDF exported and spell-checked before submission
