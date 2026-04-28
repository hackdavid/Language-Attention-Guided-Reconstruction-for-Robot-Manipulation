# Experiment C4 — Selected localization heads

**Condition ID:** `C4`  
**Goal:** Use only **low-entropy, high-magnitude** attention heads (see `experiment/03_sample_data_testing.md` T4) to build the mask.  
**Aligns with:** `experiment/02_experiment_plan.md` Step 5 — isolates effect (C).

---

## 1. What differs from C3

| Key | Value |
|-----|--------|
| `experiment.condition` | `C4` |
| `model.masking.mode` | `attention_selected` |
| `model.masking.selected_heads` | **List of head indices**, e.g. `[3, 7, 12]` — fill after diagnostic |
| `model.masking.head_selection_file` | Optional path to JSON from `scripts/select_localization_heads.py` (when implemented) |

**Config file:** `configs/experiments/C4_selected_heads.yaml`

---

## 2. Prerequisite

**Before first C4 run:** run the **localizing-head diagnostic** on the frozen backbone; save indices to `results/head_selection.json` and reference it from YAML **or** paste indices into `selected_heads`.

---

## 3. Implementation notes

- Aggregate attention **only** over `selected_heads` (and configured layers).
- Remaining pipeline identical to C3 (same `k`, same decoder, same λ unless ablating).

---

## 4. What to monitor

| Comparison | Interpretation |
|------------|----------------|
| C4 vs C3 on `val/action_loss` | C4 better → head selection is **critical** (matches localization-heads literature) |
| C4 vs C2 | C4 > C2 → **main positive result** for LA-ReconVLA-style masking |
| `metrics/aos` C4 vs C3 | Higher AOS → masks align better with object regions |

**Alignment**

- This is the **primary** condition for the paper’s scientific claim under small compute.

---

## 5. Run

```bash
python -m training.run_experiment --config configs/experiments/C4_selected_heads.yaml
python scripts/read_tracking.py --expect configs/expectations.yaml --condition C4
```
