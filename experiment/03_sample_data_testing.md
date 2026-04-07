# 03 — Sample Data Testing: Verifying the Pipeline Before Training

> **Purpose.** Before committing GPU hours, run a sequence of *cheap*
> diagnostic tests on sample data to confirm that every component works, that
> the attention signal exists, and that the training loop produces learning
> curves that move in the right direction.  Each test has a **pass/fail
> criterion** — if it fails, we diagnose before proceeding.

---

## 1  Test Hierarchy

The tests below are ordered by dependency.  Each later test assumes the
preceding ones have passed.

```
T0  Environment & VRAM budget     →  "Can we even load the model?"
T1  Data loading & shapes         →  "Is the data pipeline correct?"
T2  Forward-pass smoke test       →  "Do outputs have the right shapes?"
T3  Attention map extraction      →  "Can we read cross-attention from PaliGemma?"
T4  Localizing-head diagnostic    →  "Does the frozen backbone contain spatial signal?"
T5  Masking correctness           →  "Does top-k masking produce valid binary masks?"
T6  MAE decoder reconstruction    →  "Can the decoder reconstruct anything at all?"
T7  Loss computation & backward   →  "Do gradients flow without NaN/Inf?"
T8  Micro-training (2 epochs)     →  "Does the loss decrease on a tiny batch?"
T9  Attention evolution check     →  "Do saliency maps change during training?"
```

---

## 2  Detailed Test Specifications

### T0 — Environment & VRAM Budget

**What to run:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "google/paligemma-3b-pt-224", torch_dtype=torch.float16
).to("cuda")
print(f"Model loaded. VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
del model; torch.cuda.empty_cache()
```

**Pass criterion:** Model loads; VRAM used < 10 GB (leaving ~5 GB for decoder,
optimiser states, and batch).

**Fail diagnosis:**
- OOM → Use `load_in_4bit=True` (bitsandbytes) or switch to
  `paligemma-3b-pt-224` quantised variant.
- CUDA not available → Check runtime type (must be GPU).

---

### T1 — Data Loading & Shapes

**What to run:**
```python
from data.dataset import LIBERODataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
tasks = [
    "KITCHEN_SCENE1_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
    "KITCHEN_SCENE2_open_the_bottom_drawer_of_the_cabinet",
    "KITCHEN_SCENE3_turn_on_the_stove"
]
ds = LIBERODataset("./data/libero_spatial", tasks, max_demos_per_task=5,
                    tokenizer=tokenizer, split='train')

sample = ds[0]
print(f"Image shape:     {sample['image'].shape}")      # expect (3, 224, 224)
print(f"Input IDs shape: {sample['input_ids'].shape}")   # expect (64,)
print(f"Action shape:    {sample['action'].shape}")      # expect (7,)
print(f"Action range:    [{sample['action'].min()}, {sample['action'].max()}]")
                                                          # expect [0, 255]
print(f"Total samples:   {len(ds)}")
```

**Pass criterion:** Shapes match; action values in [0, 255]; samples > 0.

**Fail diagnosis:**
- 0 samples → HDF5 path or task name mismatch.  Print `hdf5_path.exists()`.
- Image shape wrong → Check `agentview_image` key in HDF5.
- Action range outside [0, 255] → Check `_discretise_action` clipping.

---

### T2 — Forward-Pass Smoke Test

**What to run (with a *single* sample, batch size 1):**
```python
from models.la_reconvla import LA_ReconVLA

model = LA_ReconVLA(
    backbone_name="google/paligemma-3b-pt-224",
    mask_ratio=0.25, lambda_recon=0.5, freeze_backbone=True
).to("cuda")

batch = {k: v.unsqueeze(0).to("cuda") if isinstance(v, torch.Tensor) else v
         for k, v in sample.items()}

with torch.no_grad():
    outputs = model(batch['image'], batch['input_ids'], batch['attention_mask'])

print(f"Action logits shape: {outputs['action_logits'].shape}")  # (1, 7, 256)
print(f"Recon loss:          {outputs['recon_loss'].item():.4f}")
print(f"Mask shape:          {outputs['mask'].shape if outputs['mask'] is not None else 'None'}")
print(f"Saliency shape:      {outputs['saliency'].shape if outputs['saliency'] is not None else 'None'}")
```

**Pass criterion:**
- `action_logits` shape = (1, 7, 256).
- `recon_loss` is a finite positive number.
- `mask` shape = (1, 196), values in {0, 1}, with exactly 49 ones.
- `saliency` shape = (1, 196), all values finite.

**Fail diagnosis:**
- `mask is None` → Cross-attention hooks did not fire.  PaliGemma may not
  have modules named `cross_attn`.  Print all module names and look for the
  actual attention layer names.
- `recon_loss = 0` → Mask is all zeros, or decoder received no masked tokens.
- NaN/Inf → fp16 overflow.  Try `torch.float32` for the decoder.

---

### T3 — Attention Map Extraction (Critical Diagnostic)

**What to run:**
```python
model.eval()
model._attn_maps.clear()

with torch.no_grad():
    _ = model.backbone(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        pixel_values=batch['image'].half(),
        output_attentions=True,
        return_dict=True
    )

print(f"Number of attention maps captured: {len(model._attn_maps)}")
if len(model._attn_maps) > 0:
    attn = model._attn_maps[-1]
    print(f"Attention map shape: {attn.shape}")
    # Expected: (1, num_heads, seq_len, seq_len) or similar
else:
    print("WARNING: No cross-attention maps captured.")
    print("Available module names with 'attn':")
    for name, _ in model.backbone.named_modules():
        if 'attn' in name.lower():
            print(f"  {name}")
```

**Pass criterion:** At least one attention map captured with the image-patch
dimension present (size 196 in one of the last two dims).

**Fail diagnosis:**
- No maps captured → PaliGemma uses *self-attention* (image + text tokens in
  one sequence), not separate cross-attention.  **This is expected for
  PaliGemma.**  In this case, extract self-attention and slice the
  text-to-image block:
  ```python
  # PaliGemma: tokens = [image_patches (196), text_tokens]
  # Self-attention shape: (B, H, S, S) where S = 196 + text_len
  # Cross-attention equivalent: attn[:, :, 196:, :196]
  full_attn = outputs.attentions[-1]  # last layer
  cross_equiv = full_attn[:, :, 196:, :196]  # (B, H, text_len, 196)
  ```
  Re-run T2 with this extraction method.

---

### T4 — Localizing-Head Diagnostic

**What to run:**
```python
import matplotlib.pyplot as plt
import numpy as np

# Using the cross-attention equivalent from T3
# Shape: (1, num_heads, text_len, 196)
cross_attn = cross_equiv[0]  # remove batch dim: (H, L, 196)
H = cross_attn.shape[0]

# Per-head analysis
for h in range(H):
    attn_h = cross_attn[h]  # (L, 196)
    avg_over_text = attn_h.mean(dim=0)  # (196,)

    # Spatial entropy
    p = avg_over_text / (avg_over_text.sum() + 1e-8)
    entropy = -(p * (p + 1e-8).log()).sum().item()
    max_entropy = np.log(196)

    # Magnitude (mean attention on image tokens)
    magnitude = avg_over_text.mean().item()

    print(f"Head {h:2d}: entropy={entropy:.2f}/{max_entropy:.2f}  "
          f"magnitude={magnitude:.6f}  "
          f"localizing={'YES' if entropy < 0.7 * max_entropy else 'no'}")

# Visualise top-3 lowest-entropy heads
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
sorted_heads = sorted(range(H), key=lambda h: -(cross_attn[h].mean(0) / 
                       (cross_attn[h].mean(0).sum()+1e-8) * 
                       (cross_attn[h].mean(0)+1e-8).log()).sum().item())
for i, h in enumerate(sorted_heads[:3]):
    heatmap = cross_attn[h].mean(0).reshape(14, 14).cpu().numpy()
    axes[i].imshow(heatmap, cmap='hot')
    axes[i].set_title(f'Head {h}')
plt.savefig('results/attention_maps/localizing_heads_frozen.png', dpi=150)
plt.show()
```

**Pass criterion:**
- At least 2 heads with entropy < 70% of max entropy *and* above-median
  magnitude.  These are candidate localization heads.
- At least one head's heatmap visually correlates with the target object
  location.

**Fail criterion & action:**
- All heads have entropy ≈ max entropy → Backbone has no spatial selectivity.
  This is a **show-stopper for attention masking**.  Document and fall back to
  random-mask MAE (C2).
- Heatmaps show attention on image borders/corners → Sink tokens.  Exclude
  these heads from the selected set.

**Record:** Save the list of selected head indices for use in C4.

---

### T5 — Masking Correctness

**What to run:**
```python
from models.attention_masker import AttentionGuidedMasker

masker = AttentionGuidedMasker(num_patches=196, mask_ratio=0.25, num_heads=H)
# Use the cross_equiv from T3
mask, saliency = masker(cross_equiv)

print(f"Mask shape:       {mask.shape}")         # (1, 196)
print(f"Mask sum:         {mask.sum().item()}")   # should be 49
print(f"Mask unique vals: {mask.unique().tolist()}")  # [0, 1]
print(f"Saliency range:   [{saliency.min():.6f}, {saliency.max():.6f}]")

# Visualise mask on image
mask_2d = mask[0].reshape(14, 14).cpu().numpy()
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.imshow(mask_2d, cmap='Reds')
ax.set_title(f'Attention mask (49/196 patches)')
plt.savefig('results/attention_maps/sample_mask.png', dpi=150)
```

**Pass criterion:** Exactly 49 ones in mask; values binary; saliency finite.

---

### T6 — MAE Decoder Reconstruction

**What to run:**
```python
from models.mae_decoder import MAEDecoder

decoder = MAEDecoder(embed_dim=2048, decoder_dim=256, num_patches=196).to("cuda")

# Dummy patch tokens (from backbone output)
dummy_tokens = torch.randn(1, 196, 2048, device="cuda")
dummy_mask = mask.to("cuda")

reconstructed = decoder(dummy_tokens, dummy_mask)
print(f"Reconstructed shape: {reconstructed.shape}")
# Expected: (1, 49, 16*16*3) = (1, 49, 768)
print(f"Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
print(f"Any NaN: {reconstructed.isnan().any().item()}")
```

**Pass criterion:** Shape = (1, 49, 768); no NaN; values finite.

**Fail diagnosis:**
- Shape mismatch → Check `embed_dim` matches backbone hidden size.
- NaN → Initialisation issue or numerical instability.  Check `_init_weights`.

---

### T7 — Loss Computation & Backward Pass

**What to run:**
```python
model.train()
outputs = model(batch['image'], batch['input_ids'], batch['attention_mask'])
losses = model.compute_loss(outputs['action_logits'],
                             batch['action'].unsqueeze(0).to("cuda"),
                             outputs['recon_loss'])

print(f"Total loss:  {losses['total_loss'].item():.4f}")
print(f"Action loss: {losses['action_loss']:.4f}")
print(f"Recon loss:  {losses['recon_loss']:.4f}")

# Backward
losses['total_loss'].backward()

# Check gradients
grad_norms = {}
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        grad_norms[name] = param.grad.norm().item()

print(f"\nTrainable params with gradients: {len(grad_norms)}")
print(f"Max grad norm: {max(grad_norms.values()):.6f}")
print(f"Min grad norm: {min(grad_norms.values()):.6f}")
print(f"Any NaN grads: {any(v != v for v in grad_norms.values())}")
```

**Pass criterion:**
- All three losses are finite positive numbers.
- At least 10 trainable parameters have non-zero gradients.
- No NaN gradients.
- Action loss ≈ ln(256) ≈ 5.55 at initialisation (random prediction over 256
  bins).

**Fail diagnosis:**
- NaN gradients → fp16 overflow.  Use `GradScaler` or fp32 for decoder.
- Zero gradients everywhere → Frozen layers not correctly excluded from
  `requires_grad = False`.
- Action loss << 5.55 at init → Something is wrong with loss computation
  (targets or logit shapes misaligned).

---

### T8 — Micro-Training (2 Epochs, 10 Samples)

**What to run:**

Train on a tiny subset (10 samples, 2 epochs, no gradient accumulation) to
verify the loss *decreases*.

```python
from torch.utils.data import DataLoader, Subset

tiny_ds = Subset(ds, range(min(10, len(ds))))
tiny_loader = DataLoader(tiny_ds, batch_size=2, shuffle=True)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-4
)

for epoch in range(2):
    epoch_loss = 0
    for batch in tiny_loader:
        img = batch['image'].to("cuda")
        ids = batch['input_ids'].to("cuda")
        amask = batch['attention_mask'].to("cuda")
        act = batch['action'].to("cuda")

        outputs = model(img, ids, amask)
        losses = model.compute_loss(outputs['action_logits'], act,
                                     outputs['recon_loss'])
        losses['total_loss'].backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += losses['total_loss'].item()

    print(f"Epoch {epoch}: avg loss = {epoch_loss / len(tiny_loader):.4f}")
```

**Pass criterion:** Loss at epoch 1 < loss at epoch 0.  The model is
*learning something* on this tiny batch.

**Fail diagnosis:**
- Loss increases → Learning rate too high, or loss computation error.
- Loss stays flat → Gradients not reaching trainable parameters, or the
  problem is trivially solved (check if all actions are identical).
- OOM → Reduce batch_size to 1.

---

### T9 — Attention Evolution Check

**What to run:**

After micro-training, extract the saliency map for the same sample used in T4
and compare.

```python
model.eval()
with torch.no_grad():
    outputs_post = model(batch['image'], batch['input_ids'],
                          batch['attention_mask'])
    saliency_post = outputs_post['saliency']

# Compare with pre-training saliency
cosine_sim = torch.nn.functional.cosine_similarity(
    saliency.flatten(), saliency_post.flatten(), dim=0
).item()
print(f"Saliency cosine similarity (pre vs post micro-training): {cosine_sim:.4f}")
```

**Pass criterion:** Cosine similarity < 0.99 — the saliency map *changed*
during training.  If the backbone's last 2 layers are unfrozen, attention
should have shifted at least slightly.

**Interpretation:**
- cos ≈ 1.0 → Attention did not change.  Backbone layers might still be
  frozen, or learning rate is too low for 2 epochs to matter.  Not fatal —
  may need more epochs.
- cos < 0.8 → Significant shift.  Inspect visually: did attention move
  *toward* the target object or *away*?  This is the earliest qualitative
  signal of whether the feedback loop is healthy.

---

## 3  Summary Checklist

| Test | Pass? | Notes |
|------|-------|-------|
| T0 — Environment & VRAM | | |
| T1 — Data loading | | |
| T2 — Forward pass | | |
| T3 — Attention extraction | | |
| T4 — Localizing heads | | |
| T5 — Masking correctness | | |
| T6 — MAE decoder | | |
| T7 — Loss & backward | | |
| T8 — Micro-training | | |
| T9 — Attention evolution | | |

Fill this in as you run each test.  **Do not start full training (02
experiment plan) until all T0–T8 pass.**  T9 is informational.

---

## 4  What "Learning" Looks Like (Expected Training Curves)

### 4.1  Healthy training

```
Epoch:     1    5    10   15   20
────────────────────────────────
L_total:  6.2  4.8  3.9  3.5  3.2   ← steady decrease
L_action: 5.5  4.5  3.7  3.3  3.0   ← dominates early, converges
L_recon:  1.4  0.6  0.4  0.35 0.30  ← drops fast then plateaus
AOS:      0.12 0.18 0.25 0.30 0.33  ← increases (attention sharpening)
```

**Key signals:**
- `L_action` starts near ln(256) ≈ 5.55 and decreases.
- `L_recon` decreases faster than `L_action` (reconstruction is easier).
- AOS increases monotonically (attention focuses on target over training).

### 4.2  Warning signs during training

| Observation | Likely cause | Action |
|-------------|-------------|--------|
| `L_recon` drops to near zero in epoch 1 | Reconstruction too easy (25% mask, smooth images) | Increase masking ratio or use contiguous masks |
| `L_action` does not decrease | Backbone frozen too aggressively; gradients from recon not reaching action head | Unfreeze more layers; check gradient flow |
| `L_recon` oscillates wildly | Mask changes dramatically step-to-step | Add EMA teacher (C5); reduce learning rate |
| AOS stays flat at ~0.10 | Attention is not sharpening; degenerate loop | Try warm-start; switch to selected heads |
| AOS *decreases* over training | Attention is *defocusing* — catastrophic | Stop training; diagnose. Likely degenerate loop. |
| NaN loss after N epochs | fp16 underflow or exploding gradients | Add gradient clipping; use GradScaler |
| Val loss increases while train decreases | Overfitting (expected with 150 demos) | Early stopping; report best-epoch metrics |

### 4.3  What to log and when

| What | When | Why |
|------|------|-----|
| Loss curves (all 3 components) | Every step (WandB) | Core convergence evidence |
| Saliency heatmaps (4 samples, 2 per task) | Epochs 1, 5, 10, 15, 20 | Visual evidence of attention evolution |
| Mask consistency (same image, 2 instructions) | Epochs 1, 10, 20 | Tests instruction-discrimination |
| AOS on 20 validation samples | Every 5 epochs | Quantitative grounding metric |
| Gradient norm histogram | Epochs 1, 10, 20 | Gradient health check |
| VRAM usage | Epoch 1 | Practical constraint |

---

## 5  Minimal Viable Experiment (If Time Is Extremely Limited)

If you can only run 2 conditions instead of 5:

1. **C1** (action-only baseline) — 90 min
2. **C4** (selected-head attention masking) — 90 min

Compare:
- Action accuracy: does C4 beat C1?
- AOS: does C4 attend more to target objects?
- Loss curves: does reconstruction loss drive attention sharpening?

This is the *minimum* to test the core claim.  Add C2 (random masking) if
there is any additional time — it provides the crucial "is attention better
than random?" comparison.
