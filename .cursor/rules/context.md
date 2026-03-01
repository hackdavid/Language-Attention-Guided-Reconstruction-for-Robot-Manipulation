# MASTER CONTEXT FILE — LA-ReconVLA Project
## Agent Briefing: Full Project Scope, Intent & Expected Outcomes

> **Use this file first.** Before reading PART1_WRITING_GUIDE.md or PART2_IMPLEMENTATION_GUIDE.md,  
> read this document completely. It tells you *what this project is*, *why each decision was made*,  
> and *what success looks like* for both parts.

---

## 1. WHO IS THIS FOR?

This project belongs to a student on the **MSc Artificial Intelligence programme** at the University of Roehampton. The module is **Deep Learning and Generative AI (CMP030L043)**, assessed at Level 7. The student has chosen **ReconVLA (arXiv:2508.10333)** as their selected paper — an AAAI 2026 Outstanding Paper on robotic manipulation using visual reconstruction as an auxiliary grounding task.

The student has two deliverables:

| Part | What | Words | Weight | Deadline |
|---|---|---|---|---|
| P1 | Critical Appraisal + Proposal | 2,000 | 40% | 6 March 2026 |
| P2 | Implementation Report + Code | 3,000 + GitHub | 60% | 17 April 2026 |

**Both parts must pass at 50% independently to pass the module.**

---

## 2. WHAT IS RECONVLA? (The Paper Being Critiqued)

ReconVLA is a **Vision-Language-Action (VLA)** model for robot manipulation. Here is the complete conceptual chain you must understand before writing anything:

### 2.1 The Problem It Solves

Standard VLA models (e.g., OpenVLA, RT-2) generate robot actions by feeding an image + language instruction into a large language model. The problem: **the VLM's visual attention is dispersed** — it looks at the whole scene rather than the specific object it needs to manipulate. This is called the *visual grounding problem* in robotics. When a model is told "put the black bowl in the drawer," it should focus on the bowl, not the whole kitchen.

### 2.2 ReconVLA's Solution

ReconVLA introduces an **implicit grounding paradigm**:
- It identifies a **gaze region** — the spatial area in the image corresponding to the manipulation target
- It adds a **diffusion transformer head** that is trained to *reconstruct* that gaze region from the VLM's internal visual tokens (called *reconstructive tokens*)
- The reconstruction task forces the VLM backbone to encode geometrically precise, spatially structured representations of the target region — because if the backbone doesn't "understand" the shape and position of the bowl, it can't reconstruct the gaze region
- At action time, the same backbone (now geometrically enriched) produces better action predictions

### 2.3 Why This is Novel

Before ReconVLA, grounding in VLAs worked in two ways:
- **Explicit grounding:** a separate external module (e.g., a detection model) identifies the object and passes its bounding box to the VLM — expensive, brittle
- **Chain-of-Thought (CoT) grounding:** the model outputs the bounding box as text before the action — adds latency, requires annotation

ReconVLA's implicit grounding requires **no external module and no explicit bounding box output** — the grounding happens inside the backbone through the reconstruction loss gradient.

### 2.4 Key Results Claimed

- Outperforms OpenVLA on LIBERO-Spatial, LIBERO-Long, and CALVIN benchmarks
- Shows improved attention focus via attention map visualisation
- Tested on real robot arm (qualitative results only)

---

## 3. THE VLA READING SEQUENCE — WHY IT EXISTS

The professor provided a reading sequence of 9 papers. This is not optional background — it is the **intellectual lineage** the report must demonstrate awareness of. Think of it as a staircase:

```
CLIP (2021)
  ↓ Dual encoder: image + text aligned in shared embedding space
ViLT (2021)
  ↓ Single transformer processes both modalities together
LLaVA (2023)
  ↓ Vision encoder projected into LLM token space — VLMs for general tasks
Decision Transformer (2021)
  ↓ Actions modelled as sequence tokens — bridge from VLM → robot actions
RT-2 (2023)
  ↓ VLM backbone directly outputs robot actions as text tokens
OpenVLA (2024)
  ↓ Open-source version of RT-2 style VLA, reproducible baseline
CALVIN Benchmark (2022)
  ↓ Exposes generalisation failure of behaviour cloning in long-horizon tasks
Diffusion Policy (2023)
  ↓ Diffusion model for action generation — fine-grained control advantage
ReconVLA (2025)  ← THIS IS THE PAPER BEING STUDIED
  ↓ Adds reconstruction diffusion head for gaze region — forces geometric understanding
```

**Every paper in this chain must be cited in the report's Background/Literature Review section (Part 2) and referenced selectively in the Critical Appraisal (Part 1).** The reading sequence demonstrates that the student understands *why* each architectural choice in ReconVLA was necessary — it didn't appear from nowhere.

---

## 4. THE PROPOSED IMPROVEMENT — LA-ReconVLA

### 4.1 What Was Proposed and Why

Through iterative self-critique, three candidate proposals were evaluated:

**Candidate A — Replace diffusion head with VAE**
- Rejected: VAE reconstructions are blurry; insufficient signal for geometric precision. Doesn't solve any real limitation.

**Candidate B — Replace gaze regions with attention-derived masks, keep diffusion head**
- Partially rejected: Solves gaze annotation dependency but retains inference overhead of diffusion.

**Candidate C — Replace gaze regions with attention-guided masking AND replace diffusion head with MAE decoder** ✓ SELECTED
- Addresses two independent, well-justified limitations simultaneously
- Both changes are technically grounded and feasible on free GPU
- Creates measurable hypotheses testable within compute constraints

### 4.2 The Proposed Architecture: LA-ReconVLA

**Full name:** Language-Attention Guided Masked Reconstruction VLA Model

**Core idea:** Instead of using heuristic gaze regions as reconstruction targets, extract cross-attention saliency maps from inside the VLM backbone — the patches most attended to when processing the language instruction are the semantically relevant ones. Mask those patches. Train a lightweight MAE-style transformer decoder (single forward pass) to reconstruct them.

**Architecture components:**

```
Input: Image (224×224) + Language Instruction
         ↓
   PaliGemma-3B Backbone (partially frozen)
         ↓                    ↓
  [Image patch tokens]   [Cross-attention maps]
   (196 patches)           (B, H, L, P)
         ↓                    ↓
                    AttentionGuidedMasker
                    - Weighted average over heads
                    - Average over language tokens  
                    - Top-k threshold (k = 25% × 196 = 49 patches)
                    → binary mask (B, 196)
                         ↓
              MAEDecoder (4-layer transformer)
              - Projects patch tokens to 256-dim
              - Replaces masked positions with learnable mask token
              - Single forward pass reconstruction
              → L_recon = MSE(predicted, original pixels)
                         ↓
              ActionHead (2-layer MLP)
              → L_action = CrossEntropy(logits, action_bins)
                         ↓
         L_total = L_action + λ · L_recon
```

**Key design numbers:**
- Masking ratio: top 25% attended patches (49 of 196)
- MAE decoder: 4 transformer layers, 256 hidden dim, 8 heads
- λ (recon loss weight): 0.5 (ablated: 0.1, 0.5, 1.0)
- Action discretisation: 256 bins per DoF, 7 DoF = 7 × 256 classification tasks

### 4.3 Why This Is Technically Sound

1. **Language attention maps are semantically grounded:** Cross-attention in transformers literally measures "how much does this language token attend to this image patch?" — aggregating this signal gives the patches most relevant to the instruction, which is a principled proxy for gaze.

2. **MAE decoder is sufficient:** The reconstruction task only needs to be strong enough to force the backbone to encode spatial structure. MAE (He et al., 2022) achieves this without requiring photorealistic outputs. Single-pass reconstruction = no inference overhead from iterative denoising.

3. **Gradient flow is cleaner:** The MAE decoder provides direct gradient signals to the backbone encoder in one pass. Diffusion requires gradients to traverse T timesteps — potentially causing gradient vanishing or dilution.

4. **Information bottleneck principle:** Masking high-attention patches and requiring reconstruction creates a bottleneck — the model must retain spatial information it would otherwise discard. This regularisation effect is well-studied in self-supervised learning theory.

---

## 5. THE THREE HYPOTHESES

All experiments in Part 2 are designed to test these three hypotheses, which were stated in Part 1:

| Hypothesis | Statement | How Tested |
|---|---|---|
| **H1** | LA-ReconVLA achieves within 5% of ReconVLA's task success rate on LIBERO-Spatial without using gaze annotations | Task Success Rate (TSR) in simulation rollouts |
| **H2** | LA-ReconVLA has 3–5× lower inference latency than ReconVLA due to MAE vs diffusion decoder | Latency benchmark over 100 forward passes |
| **H3** | LA-ReconVLA shows higher attention concentration on task-relevant objects vs baseline VLA without reconstruction | Attention Overlap Score (AOS): IoU between top-25% attention patches and GT object bounding box |

**H4 (aspirational):** LA-ReconVLA generalises better to out-of-distribution instructions where gaze annotations would be unavailable. This is mentioned in Part 1 but not fully tested in Part 2 (acknowledged as a limitation).

---

## 6. COMPUTE CONSTRAINTS AND DESIGN DECISIONS

The entire implementation is designed for **free-tier Colab (T4 GPU, 15GB VRAM)** or **Kaggle (P100 GPU)**. Every technical decision was made with this constraint in mind:

| Constraint | Decision Made | Why |
|---|---|---|
| 15GB VRAM | PaliGemma-3B (not 7B) backbone | 3B fits in fp16 with room for decoder |
| Limited VRAM | Freeze most of backbone (fine-tune last 2 layers only) | Reduces gradient memory |
| Limited VRAM | `torch.float16` + GradScaler | Halves activation memory |
| Limited VRAM | Batch size 8 + gradient accumulation × 4 = effective batch 32 | Stable training without OOM |
| Session limits | 3 tasks × 50 demos × 20 epochs ≈ 90 mins | Stays within Colab session |
| No robot hardware | LIBERO-Spatial simulation benchmark | Reproducible, no hardware needed |
| No gaze annotations | Attention-guided masking | This is the research contribution itself |

**If 3B model still causes OOM:** Load with `load_in_4bit=True` (bitsandbytes) — adds 3-4% performance drop, acceptable for proof-of-concept.

---

## 7. DATASET CONTEXT

**LIBERO-Spatial** is the primary dataset. Here is why it was chosen over alternatives:

| Dataset | Reason Considered | Decision |
|---|---|---|
| LIBERO-Spatial | Used in ReconVLA evaluation; free; small; object rearrangement tasks (good for attention analysis) | **Selected** |
| LIBERO-Long | Long-horizon task chains; better stress test but requires more compute | Mentioned in report as future work |
| CALVIN | Used in ReconVLA; requires MuJoCo setup; harder to run on Colab | Referenced for comparison, not trained on |
| BridgeData V2 | Real robot data; large (~130GB); not feasible | Referenced as scale context |

**Subset used for training:** 3 tasks × 50 demos = ~150 demonstrations, ~4,500–7,500 timesteps. This is intentionally small — the goal is **proof-of-concept demonstration**, not state-of-the-art performance. The report explicitly acknowledges this as a limitation.

---

## 8. THE FOUR EXPERIMENTAL CONDITIONS

Every claim in the results section must be backed by one of these four conditions run side-by-side:

```
Condition 1: BASELINE
  - Standard VLA (backbone + action head, no reconstruction)
  - Purpose: Lower bound on performance
  - Shows: How much the reconstruction task helps at all

Condition 2: LA-ReconVLA RANDOM MASKING (Ablation 1)
  - Attention masking replaced with random patch masking
  - MAE decoder kept
  - Purpose: Isolates the contribution of attention-guided masking
  - Tests: "Does it matter WHERE we mask, or just that we mask?"

Condition 3: LA-ReconVLA λ ABLATION (Ablation 2)
  - Run with λ ∈ {0.1, 0.5, 1.0}
  - Purpose: Sensitivity analysis on reconstruction loss weight
  - Tests: Is the improvement fragile to hyperparameter choice?

Condition 4: LA-ReconVLA FULL (Proposed Method)
  - Attention masking + MAE decoder + λ = 0.5
  - Purpose: The proposed contribution
  - Expected to outperform Conditions 1 and 2 on TSR and AOS
```

---

## 9. WHAT COUNTS AS SUCCESS

### For Part 1 (Distinction ≥ 70%)
- [ ] Critique identifies **non-obvious** limitations (gaze annotation circular dependency, missing ablation, data leakage — not just "needs more data")
- [ ] Proposal names a **specific architecture change** with a loss function written out mathematically
- [ ] Proposal is **directly causally linked** to the critique (the improvement fixes the identified flaw)
- [ ] Hypotheses are **measurable** with specific numbers (not "the model will be better")
- [ ] Reading sequence papers are cited to demonstrate field awareness
- [ ] Word count: 1,900–2,100 words

### For Part 2 (Distinction ≥ 70%)
- [ ] Code is **functional, modular, and reproducible** — runs top-to-bottom in Colab
- [ ] README has exact reproduction commands
- [ ] **At least 3 ablation conditions** reported (not just baseline vs proposed)
- [ ] Results include both **quantitative metrics** (TSR, AOS, latency) and **qualitative visualisations** (attention maps)
- [ ] Discussion section connects back to H1–H3: did the hypotheses hold?
- [ ] Limitations are **acknowledged honestly** (small dataset, simulation only, frozen backbone noise)
- [ ] Ethics section addresses specific risks: LIBERO's limited visual distribution, latency constraints for real-time deployment, sim-to-real transfer risks
- [ ] Word count: 2,800–3,200 words

### For the Code (GitHub)
- [ ] All modules are in separate files (not one giant notebook cell)
- [ ] `configs/train_config.yaml` centralises all hyperparameters
- [ ] Training script logs to WandB
- [ ] Evaluation script produces all reported metrics programmatically
- [ ] Results saved to `results/` folder (not hardcoded print statements)
- [ ] README includes a results table with actual numbers

---

## 10. KNOWN RISKS AND MITIGATIONS

These are challenges the agent and student should anticipate:

| Risk | Likelihood | Mitigation |
|---|---|---|
| Cross-attention maps from frozen backbone are noisy | High | Aggregate across last 3 layers, not just last 1; use weighted head aggregation |
| PaliGemma-3B cross-attention structure differs from expected | Medium | Fall back to self-attention over image tokens if cross-attn not accessible; document in report |
| Attention masking target isn't always on the manipulation object | Medium | This is expected — acknowledged as a limitation; AOS score measures this directly |
| LIBERO env setup fails on Colab | Low | Use RLDS/HDF5 files directly without simulation env (offline training on demonstrations) |
| TSR can't be measured without simulation rollout | Medium | Report offline action prediction accuracy as proxy; note this limitation clearly |
| MAE decoder reconstruction is low quality | Low | Quality doesn't need to be high — only sufficient gradient signal matters; show examples anyway |

---

## 11. FILE READING ORDER FOR THE AGENT

When using these guides to write or implement, always read in this order:

```
1. THIS FILE (MASTER_CONTEXT.md)         ← Full understanding of the project
2. PART1_WRITING_GUIDE.md               ← Section-by-section writing instructions for P1
3. PART2_IMPLEMENTATION_GUIDE.md        ← Step-by-step code + report instructions for P2
```

Never write Part 1 or Part 2 content without having read this master context file first. The guides assume this context is already loaded.

---

## 12. QUICK REFERENCE — ALL KEY CITATIONS

| Paper | Key Concept | Where to Cite |
|---|---|---|
| Song et al. (2025) arXiv:2508.10333 — ReconVLA | Primary paper | P1 everywhere; P2 intro + background |
| Radford et al. (2021) ICML — CLIP | Vision-language alignment foundation | P1 §2.1; P2 background |
| Kim et al. (2023) arXiv:2304.08485 — LLaVA | VLM backbone architecture | P1 §2.1; P2 background + methodology |
| Brohan et al. (2023) arXiv:2307.15818 — RT-2 | First VLA, text-token actions | P1 §2.4; P2 background |
| Kim et al. (2024) arXiv:2406.09246 — OpenVLA | Direct baseline comparison | P1 §2.4; P2 experiments |
| Chi et al. (2023) arXiv:2303.04137 — Diffusion Policy | Diffusion for actions; latency baseline | P1 §2.3; P2 background |
| He et al. (2022) CVPR — MAE | Justifies MAE decoder choice | P1 §3.3; P2 methodology |
| Ho et al. (2020) NeurIPS — DDPM | ReconVLA's diffusion head foundation | P1 §2.1 + §2.3 |
| Mees et al. (2022) arXiv:2112.03227 — CALVIN | Benchmark used in evaluation | P1 §2.4; P2 experiments |
| Chen et al. (2021) arXiv:2106.01345 — Decision Transformer | Actions as sequence tokens | P2 background |
| Dosovitskiy et al. (2021) — ViLT arXiv:2102.03334 | Single-transformer multimodal processing | P2 background |
