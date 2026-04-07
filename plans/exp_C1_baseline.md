# Experiment C1 — Action-only baseline (VLA without reconstruction)

**Condition ID:** `C1`  
**Goal:** Establish a **lower bound**. No MAE decoder, no masking, no reconstruction loss.  
**Aligns with:** `experiment/02_experiment_plan.md` Step 2; comparison anchor for C2–C5.

---

## 1. What differs from base config

| Key | Value |
|-----|--------|
| `experiment.condition` | `C1` |
| `model.reconstruction.enabled` | `false` |
| `model.reconstruction.lambda_recon` | `0` |
| `model.masking.mode` | `none` |

**Config file:** `configs/experiments/C1_action_only.yaml`

---

## 2. Data: what to load

- Same **3 tasks × 50 demos** as all conditions (`configs/base.yaml` → `data.tasks`, `data.max_demos_per_task`).
- Source: HuggingFace snapshot **or** HDF5 — unchanged from base; only the **model loss** changes.

**HuggingFace snippet** (inspect / iterate; full loader lives in `data/` when implemented):

```python
from datasets import load_dataset

# Streaming: good for a quick count / schema check
ds = load_dataset(
    "openvla/modified_libero_rlds",
    "libero_spatial_no_noops",
    split="train",
    streaming=True,
)
sample = next(iter(ds))
# Keys depend on RLDS schema; map to image, language, action in dataset.py
```

---

## 3. Implementation checklist

1. Load `configs/experiments/C1_action_only.yaml` (merges over `base.yaml`).
2. Build model **without** `MAEDecoder` and **without** attention masker in the loss path.
3. Optimise **only** `L_action` (cross-entropy on discretised 7×256).
4. Log metrics (see §5).

---

## 4. What to monitor (success / failure)

| Signal | Healthy | Concerning |
|--------|---------|------------|
| `train/action_loss` | Decreases over epochs | Flat or increasing from start |
| `val/action_loss` | Tracks train without huge gap | Val ≫ train (overfit) or val flat |
| `train/total_loss` | Same as `train/action_loss` | Any recon component non-zero → bug |
| Grad norms | Finite, not exploding | NaN / Inf |

**Alignment with research goal**

- C1 **does not** test the LA-ReconVLA hypothesis; it **isolates** whether later gains come from reconstruction vs other noise.
- If C2+ cannot beat C1 on action accuracy / TSR, the auxiliary path is not helping at this scale.

---

## 5. Logging keys (W&B + MLflow)

Standard keys (see `training/trackers.py`):

- `train/action_loss`, `val/action_loss`, `train/total_loss`, `val/total_loss`
- `epoch`, `lr`, `global_step`
- `config/condition` = `C1`

---

## 6. How to run

```bash
python -m training.run_experiment --config configs/experiments/C1_action_only.yaml
python scripts/read_tracking.py --expect configs/expectations.yaml --condition C1
```

---

## 7. Expected narrative in the paper

> We train a discrete 7-DoF action head on top of a partially frozen PaliGemma backbone without any reconstruction auxiliary, matching the standard VLA fine-tuning baseline.
