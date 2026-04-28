# Experiment C2 — Random-mask MAE + action

**Condition ID:** `C2`  
**Goal:** Test whether **any** masked reconstruction (not language-guided) improves the backbone for actions.  
**Aligns with:** `experiment/02_experiment_plan.md` Step 3 — isolates effect (A).

---

## 1. What differs from C1

| Key | Value |
|-----|--------|
| `experiment.condition` | `C2` |
| `model.reconstruction.enabled` | `true` |
| `model.reconstruction.lambda_recon` | `0.5` (or override in YAML) |
| `model.masking.mode` | `random` |
| `model.masking.mask_ratio` | `0.25` |

**Config file:** `configs/experiments/C2_random_mae.yaml`

---

## 2. Implementation notes

- **Mask:** sample exactly `k = floor(mask_ratio * num_patches)` patch indices **uniformly at random** per forward pass (or per sample per epoch — document choice).
- **Decoder:** 4-layer MAE-style decoder; **single forward** reconstruction loss on masked patches.
- **Loss:** `L_total = L_action + lambda_recon * L_recon` (MSE in pixel or normalised patch space — keep consistent across C2–C5).

**Code touchpoint (conceptual)**

```python
# Random mask: no attention tensors
mask = random_topk_mask(batch_size, num_patches=196, k=49, device=device)
```

---

## 3. What to monitor

| Metric | Favours hypothesis | Interpretation |
|--------|-------------------|----------------|
| `val/action_loss` vs C1 | Lower than C1 | Generic reconstruction helps regularise representations |
| `val/action_loss` vs C1 | ≈ C1 | Reconstruction not useful at this scale / too easy |
| `train/recon_loss` | Decreases quickly then plateaus | Expected; watch if → 0 instantly (task too easy) |
| `val/recon_mse` | Low but action also improves | Good |
| `val/recon_mse` | Very low, action flat | Decoder shortcuts; see `experiment/01_theoretical_analysis.md` §3.3 |

**Alignment**

- If C2 > C1: auxiliary reconstruction is a **valid pressure**; C3–C4 test if **where** you mask matters.
- If C2 ≈ C1: before claiming attention helps, you may need **harder masking** (A2 ablation).

---

## 4. Logging keys

All C1 keys plus:

- `train/recon_loss`, `val/recon_loss` (or `val/recon_mse`)
- `config/masking_mode` = `random`

---

## 5. Run

```bash
python -m training.run_experiment --config configs/experiments/C2_random_mae.yaml
python scripts/read_tracking.py --expect configs/expectations.yaml --condition C2
```
