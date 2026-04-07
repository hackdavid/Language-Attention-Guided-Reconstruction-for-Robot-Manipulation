# Experiment C5 — Selected heads + EMA teacher for masks

**Condition ID:** `C5`  
**Goal:** Stabilise pseudo-labels by computing **masks from an EMA copy** of the backbone (no gradient through mask path).  
**Aligns with:** `experiment/02_experiment_plan.md` Step 6.

---

## 1. What differs from C4

| Key | Value |
|-----|--------|
| `experiment.condition` | `C5` |
| `model.masking.mode` | `attention_selected` |
| `model.ema.enabled` | `true` |
| `model.ema.decay` | `0.999` (try `0.99` if too slow) |
| `model.masking.mask_source` | `ema_teacher` |

**Config file:** `configs/experiments/C5_ema_teacher.yaml`

---

## 2. Implementation notes

Each training step:

1. Update **student** weights with usual optimizer.
2. EMA: `θ_ema ← β θ_ema + (1-β) θ_student` (on backbone subset that affects attention).
3. **Forward for mask:** run **teacher** forward (eval mode, `torch.no_grad()`), extract attention, build mask.
4. **Forward for loss:** student uses that fixed mask for reconstruction + action loss as usual.

---

## 3. What to monitor

| Signal | Meaning |
|--------|---------|
| C5 better than C4 on val | Mask instability was hurting training |
| C5 ≈ C4 | EMA not needed at this scale |
| C5 worse | EMA too slow (try lower β) or implementation bug |
| `mask/delta_kl_epoch` | Should decrease over time if stabilising |

---

## 4. Run

```bash
python -m training.run_experiment --config configs/experiments/C5_ema_teacher.yaml
python scripts/read_tracking.py --expect configs/expectations.yaml --condition C5
```
