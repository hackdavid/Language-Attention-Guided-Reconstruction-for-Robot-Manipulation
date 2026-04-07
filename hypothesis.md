# LA-ReconVLA — Experimental Plan (from Part 1 report)

**Source:** `reference/A00084632(Daud-ibrahim-Dewan).pdf` — *Language-Attention Guided Masked Reconstruction for Vision-Language-Action Models* (CMP-L043-0 Part 1 Assessment Report, Student A00084632).

This document distils the **proposed method** and **Part 2 empirical plan** described in that report so implementation and evaluation can follow one checklist.

---

## 1. Research question

Can **gaze-region supervision** in ReconVLA-style training be replaced by **language-driven attention masking** for reconstruction targets, while replacing the **diffusion transformer** decoder with a **single-pass MAE decoder**, without sacrificing task performance and with lower inference latency?

---

## 2. Method summary: LA-ReconVLA

**Full name:** Language-Attention Guided Masked Reconstruction VLA.

| Aspect | ReconVLA (baseline concept) | LA-ReconVLA (proposed) |
|--------|----------------------------|-------------------------|
| Reconstruction target | Gaze region (annotated) | Top-*k* attention patches from cross-attention |
| Decoder | Diffusion transformer | 4-layer MAE decoder |
| Inference | *T* denoising steps | Single forward pass |
| External annotations | Gaze / eye-tracking | None |

**Backbone:** PaliGemma-3B (partially frozen; report suggests last layers fine-tuned — see hyperparameters).

**Inputs:** 224×224 RGB image + language instruction.

### Pipeline (high level)

1. **Cross-attention maps** — From the VLM, aggregate attention between language tokens and image patch tokens; aggregate across heads and language tokens; optionally use last 3 layers (weighted) to reduce noise from frozen layers.
2. **Attention-guided masking** — Saliency map *A* over *P* = 196 patch positions. Top-*k* with *k* = 49 (25% of patches). Binary mask *M*; replace masked tokens with learnable `[MASK]` before the decoder.
3. **MAE decoder** — 4-layer transformer (hidden dim 256, 8 heads) reconstructs pixel values at masked positions in **one** forward pass. Loss: MSE between decoded and original pixels at masked patches.
4. **Joint training** — \(L_{\text{total}} = L_{\text{action}} + \lambda \cdot L_{\text{recon}}\).  
   - \(L_{\text{action}}\): cross-entropy over discretised actions (7 DoF × 256 bins per DoF).  
   - \(\lambda\): default **0.5**; ablations **0.1, 0.5, 1.0**.

**Failure-mode note (from report):** If cross-attention is noisy, aggregate across last 3 layers or fall back to self-attention over image tokens; document the choice in Part 2.

---

## 3. Hypotheses (to test in Part 2)

| ID | Hypothesis | Null H₀ | How to test (per report) |
|----|------------|---------|---------------------------|
| **H1** | Task success within **5%** of ReconVLA on LIBERO-Spatial **without** gaze annotations | No significant difference | Simulation rollouts |
| **H2** | **3×–5×** lower inference latency vs ReconVLA (single-pass MAE vs diffusion) | No significant latency reduction | 100 forward-pass latency benchmark |
| **H3** | Higher **attention concentration** on task-relevant objects vs VLA without reconstruction | No difference in AOS | **Attention Overlap Score (AOS):** IoU between top 25% attention patches and GT object box |
| **H4** | Robustness on **out-of-distribution** instructions where gaze is unavailable | — | Aspirational; full testing may be deferred |

---

## 4. Part 2 experimental conditions

Implement on **LIBERO-Spatial** with **constrained compute** (report: Colab **T4** GPU).

**Training scale (report):** 3 tasks × 50 demos.

**Conditions to report:**

1. Baseline VLA (no reconstruction auxiliary).
2. LA-ReconVLA with **random masking** (ablation: is gain from masking generally vs attention-guided masking).
3. LA-ReconVLA with **λ ablation** (e.g. 0.1, 0.5, 1.0).
4. LA-ReconVLA **full** (attention-guided masking + chosen λ).

Provide **code, configs, and random seeds** for reproducibility.

---

## 5. Hyperparameters (Part 2 target)

| Parameter | Value | Notes |
|-----------|--------|--------|
| Image size | 224×224 | ViT-style |
| Patch count | 196 | 14×14 grid |
| Top-*k* / masking | *k* = 49 (25%) | Most-attended patches |
| MAE decoder | 4 layers | Hidden 256, 8 heads |
| λ (recon weight) | 0.5 | Ablations: 0.1, 0.5, 1.0 |
| Action discretisation | 256 bins per DoF | 7 DoF total |
| Batch size | 8 | Gradient accumulation ×4 |
| Backbone | PaliGemma-3B | Report: last **2** layers fine-tuned (Appendix B) |

---

## 6. Ethics and scope (from report)

- Simulation data (**LIBERO**); no human/animal data in the described work.
- Real deployment of VLAs requires latency and safety evaluation beyond simulation.

---

## 7. References (IEEE-style list from report)

The Part 1 report cites ReconVLA [1], OpenVLA [2], RT-2 [3], MAE [4], DDPM [5], CLIP-style alignment [6], LLaVA [7], Diffusion Policy [8], CALVIN [9], Decision Transformer [10], ViLT [11], information bottleneck [12]. Full bibliographic entries are in the PDF (pages IX–X).

---

## 8. Appendices in source PDF

- **Appendix A:** Gantt chart (Matplotlib).
- **Appendix B:** Hyperparameter summary (included in §5 above).
