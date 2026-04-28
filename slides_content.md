# LA-ReconVLA: Language-Attention Guided Masked Reconstruction for Vision-Language-Action Models

## Slide Deck Content (Storytelling Flow)

---

### Slide 1  Title

**LA-ReconVLA**
Language-Attention Guided Masked Reconstruction for VLA Models

- Removing the annotation bottleneck in robot learning
- Daud Ibrahim Dewan | Student ID: A00084632
- University of Roehampton, London
- Module: CMP-L043-0 Deep Learning

---

### Slide 2  How Do Robots Learn to Do Tasks?

> Imagine teaching a robot to pick up a lemon from a table.

1. **Show it examples**  A human demonstrates the task many times (like showing a child how to hold a cup)
2. **Record everything**  Cameras capture what the robot "sees"; sensors record what its arm "does"
3. **Learn the pattern**  The robot's brain (a neural network) finds the connection between what it sees and what action to take
4. **Repeat on its own**  Given a new scene, the robot predicts the correct action

**The key question:** How does the robot know *which part of the scene* to focus on?

---

### Slide 3  The Traditional Approach: Object Detection

The most common pipeline in robot learning today:

```
Camera Image --> Object Detector --> Find "bowl" --> Plan Action --> Move Robot
```

1. An **object detection model** (like YOLO or Grounding DINO) draws bounding boxes around objects
2. The robot identifies the target object (e.g., "black bowl") from the detected boxes
3. Action policy uses the detected object's position to compute robot movements

**Works well when:**
- Objects are clearly visible and well-defined
- There are plenty of labelled training data for detection
- The environment is clean and predictable

---

### Slide 4  Issues With the Object Detection Approach

| Problem | Why It Matters |
|---------|---------------|
| **Annotation cost** | Every training image needs labelled bounding boxes  expensive and slow |
| **Domain gap** | Detectors trained on internet images struggle with robot camera angles, lighting, and clutter |
| **Brittle to novel objects** | If the robot encounters an object the detector has never seen, it fails completely |
| **Ignores task context** | Detection finds "a bowl" but doesn't understand "pick up *that* bowl and put it *in the drawer*" |
| **Separate pipeline** | Detection and action are trained separately  errors compound across stages |

**The deeper problem:** Object detection treats everything as a box-finding exercise. But manipulation is about *what to do with* the object, not just *where* it is.

---

### Slide 5  How Real Robots Actually Work (7 Degrees of Freedom)

A robot arm doesn't just "move to a position." It must predict 7 continuous values simultaneously:

| Dimension | What It Controls | Example |
|-----------|-----------------|---------|
| x | Left-right position | Move 10cm to the right |
| y | Forward-backward position | Reach forward 5cm |
| z | Up-down position | Lift arm 15cm |
| roll | Rotation around x-axis | Tilt wrist sideways |
| pitch | Rotation around y-axis | Tilt wrist forward |
| yaw | Rotation around z-axis | Rotate wrist left/right |
| gripper | Open or close | 0 = fully open, 1 = fully closed |

**The challenge:** Each of these 7 values must be predicted accurately *at every timestep*  that's 256 possible bins per dimension, per step.

A VLA (Vision-Language-Action) model takes an image + language instruction and outputs all 7 values in one pass.

---

### Slide 6  What Is ReconVLA and Why Does It Exist?

**ReconVLA** (Song et al., 2025) rethinks how robots ground language to visual scenes.

**Core insight:** Instead of detecting objects with bounding boxes, force the model to *reconstruct* the region it should focus on.

**How it works:**
1. Identify a **gaze region** (where a human would look during the task)
2. Mask that region in the model's internal representation
3. Train a **reconstruction decoder** to recover the masked pixels
4. This forces the backbone to encode detailed information about that region

**Why this is better than detection:**
- The model learns *spatial understanding*, not just box coordinates
- Reconstruction pressure improves the internal representation (information bottleneck principle)
- The backbone learns geometry, shape, and position of the target  richer than a bounding box

---

### Slide 7  The Bottleneck: Data Annotation + Gaze Regions

ReconVLA works  but at a cost:

**The annotation pipeline:**
1. ReconVLA requires **gaze region annotations**  identifying where in the image the manipulation target is
2. In practice, it uses **Grounding DINO** (an object detector) to automatically generate these regions
3. This creates a preprocessing pipeline on **100k+ trajectories, 2M+ samples**

**Why this is a problem:**
- **Scalability:** Every new task/environment needs fresh annotation or a reliable detector
- **Cost:** Grounding DINO itself is a heavy model requiring compute
- **Circular dependency:** You need a good detector to train a good robot policy  but if the detector fails, the robot fails
- **Accessibility:** Small research labs cannot afford this preprocessing overhead

**The research question:** Can we remove this annotation bottleneck entirely?

---

### Slide 8  The Reconstruction Concept: Replacing Object Detection

**Key idea:** The model already knows where to look  we just need to extract that knowledge.

**Three converging lines of evidence:**

| Evidence | Source | What It Shows |
|----------|--------|---------------|
| Reconstruction forces grounding | ReconVLA [1], MAE [4] | If you mask a region and force reconstruction, the backbone must encode that region's details |
| Grounding emerges without labels | GroundLMM [13] | Large VLMs develop grounding ability *without* explicit supervision |
| Only a few heads localize | Localization Heads [14] | A small number of attention heads consistently attend to task-relevant objects |

**The substitution:**
- Instead of an external gaze annotation, use the model's **own cross-attention** to identify the important region
- Instead of a heavy diffusion decoder, use a lightweight **MAE (Masked Autoencoder) decoder**

---

### Slide 9  Our Approach: LA-ReconVLA and Why We Chose It

**LA-ReconVLA = Language-Attention guided ReconVLA**

We replace two components of ReconVLA:

| Aspect | ReconVLA (original) | LA-ReconVLA (ours) |
|--------|--------------------|---------------------|
| Region selection | Gaze annotation + Grounding DINO | Cross-attention saliency from the model itself |
| Reconstruction decoder | Diffusion Transformer (50-1000 steps) | MAE decoder (4 layers, single forward pass) |
| External dependencies | Object detector, annotation pipeline | None |
| Inference | Multi-step denoising | One-shot prediction |

**Why this should work (with evidence):**

1. **GroundLMM [13]:** Grounding ability emerges in VLMs without explicit supervision  the signal is already in the attention maps
2. **Localization Heads [14]:** Only a few attention heads act as consistent object localizers  we select these, not average all heads
3. **MAE [4]:** Single-pass reconstruction provides cleaner gradients than multi-step diffusion (gradient path depth: O(4) vs O(T * D))
4. **SemMAE/SemMIM [19,21]:** Text-guided masking is better than random masking for cross-modal alignment

**The strongest advantage: zero external annotations needed.**

---

### Slide 10  How LA-ReconVLA Works (Pipeline)

```
                    Input
                      |
            +---------+---------+
            |                   |
         Image (224x224)    Language Instruction
            |                   |
            +--------+----------+
                     |
              PaliGemma-3B Backbone
              (frozen, last 2 layers fine-tuned)
                     |
          +----------+-----------+
          |                      |
    Cross-Attention          Patch Tokens
    Maps (lang -> img)       (256 patches)
          |                      |
    Aggregate over           Identify top-25%
    selected heads           most-attended patches
          |                      |
          +----------+-----------+
                     |
              Binary Mask M
              (49/256 patches)
                     |
              Replace masked tokens
              with learnable [MASK]
                     |
              MAE Decoder (4 layers)
              Reconstructs masked pixels
                     |
              L_recon = MSE(reconstructed, original)
                     |
              L_total = L_action + lambda * L_recon
```

**The self-reinforcing loop:**
- Better attention → better masks → better reconstruction → better backbone representations → even better attention

---

### Slide 11  Experiment Plan: Five Conditions

We designed 5 conditions, each varying exactly one dimension:

| ID | Condition | Mask Source | Decoder | Key Question |
|----|-----------|-------------|---------|-------------|
| **C1** | Action-only baseline | None | None | Lower bound: no reconstruction |
| **C2** | Random-mask MAE | Random 25% patches | 4-layer MAE | Does *any* reconstruction help? |
| **C3** | Naive attention MAE | All-head avg cross-attn, top-25% | 4-layer MAE | Does naive attention beat random? |
| **C4** | Selected-head MAE | Localization-head cross-attn, top-25% | 4-layer MAE | Does head selection matter? |
| **C5** | Selected-head + EMA teacher | EMA backbone attention, top-25% | 4-layer MAE | Does mask stabilisation help? |

**Shared setup:**
- Backbone: PaliGemma-3B (frozen, last 2 layers fine-tuned)
- Dataset: LIBERO-Spatial, 3 tasks x 50 demonstrations
- Training: 3 epochs, batch size 8-16, AdamW lr=1e-4
- Hardware: Kaggle P100 / Colab T4

---

### Slide 12  Results So Far: Not Yet As Expected

**Experimental results (3 epochs, single seed 42):**

| Condition | Action Loss | Recon Loss | Val Action MAE (mean) | Val Total Loss |
|-----------|------------|------------|----------------------|----------------|
| **C1** (Baseline) | 0.234 |  | 0.808 | 1.026 |
| **C2** (Random mask auto-encoder MAE) | 0.244 | 0.046 | 0.805 | 1.067 |
| **C3** (Naive Attn) | 0.244 | 0.048 | 0.809 | 1.085 |
| **C4** (Selected Heads) | 0.245 | 0.052 | 0.809 | 1.086 |
| **C5** Exponential Moving Average (EMA Teacher) | 0.245 | 0.052 | 0.809 | 1.084 |

**What we observe:**
- Reconstruction loss decreases (the decoder is learning) but **action loss does not improve** over baseline
- All conditions converge to nearly identical val MAE (~0.809)
- The gap between conditions is within noise margin

**Why  likely reasons:**
1. **Insufficient training:** Only 3 epochs with 150 demos  the model barely starts learning
2. **Backbone too frozen:** Last 2 layers out of 26+ layers provide limited gradient flow
3. **Reconstruction too easy:** 25% masking with 75% context  decoder solves by interpolation, backbone gets no useful signal
4. **Compute constraint:** T4/P100 with batch_size 6-8 limits the effective training signal per step

---

### Slide 13  Further Experiment Plan: Fine-Tuning with LoRA

**The next step: unlock the backbone with LoRA (Low-Rank Adaptation)**

**What LoRA does:**
- Injects small trainable matrices (rank-16) into each attention and MLP layer of the backbone
- Instead of fine-tuning only 2 layers, LoRA adapts *all* layers efficiently
- Trainable parameters increase from ~2M to ~15M while keeping memory manageable

**Why LoRA should work for LA-ReconVLA (with evidence):**

| Evidence | Source | Relevance |
|----------|--------|-----------|
| LoRA on VLMs preserves pre-trained knowledge while enabling adaptation | Hu et al., 2022 [LoRA] | We need the backbone's grounding signal but also need to adapt it for robot tasks |
| LoRA enables fine-tuning LLMs on single GPUs | Official LoRA paper benchmarks | Fits our T4/P100 constraint |
| Our frozen backbone produces stale attention maps | Current experiment results | LoRA allows attention patterns to evolve during training |
| LoRA ranks 8-16 match full fine-tuning on similar VLM tasks | Empirical studies on PaliGemma | rank=16 with alpha=32 is well-validated |

**Planned LoRA conditions:**

| Config | LoRA Target | r | alpha | Dropout |
|--------|------------|---|-------|---------|
| C3_lora | q/k/v/o_proj + MLP (vision + language) | 16 | 32 | 0.05 |
| C5_lora | Same + EMA teacher | 16 | 32 | 0.05 |

**Expected improvement:**
- Attention patterns should become instruction-conditioned (currently near-static)
- Reconstruction loss should create meaningful gradient flow into the backbone
- Action MAE should decrease as the backbone learns to encode task-relevant spatial information

---

### Slide 14  Questions?

**Summary of key points:**
1. Robot manipulation needs spatial grounding  not just object detection
2. ReconVLA showed reconstruction pressure forces grounding, but requires expensive annotations
3. LA-ReconVLA proposes replacing annotations with the model's own attention + replacing diffusion with MAE
4. Current experiments show the pipeline works but the frozen backbone limits learning
5. LoRA fine-tuning is the next step to unlock the backbone's full potential

**Key references:**
- [1] Song et al., "ReconVLA," arXiv:2508.10333, 2025
- [4] He et al., "Masked Autoencoders," CVPR, 2022
- [13] Zhao et al., "GroundLMM," arXiv:2410.08209, 2024
- [14] Zhang et al., "Localization Heads," arXiv:2503.06287, 2025

Thank you. Questions?

---
