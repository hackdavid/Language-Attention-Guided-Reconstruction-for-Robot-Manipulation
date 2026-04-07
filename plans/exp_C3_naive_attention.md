# Experiment C3 — Naive attention-guided masking (all heads)

**Condition ID:** `C3`  
**Goal:** Test whether **raw** language-conditioned attention maps, averaged across heads, yield better masks than random.  
**Aligns with:** `experiment/02_experiment_plan.md` Step 4 — isolates effect (B).

---

## 1. What differs from C2

| Key | Value |
|-----|--------|
| `experiment.condition` | `C3` |
| `model.masking.mode` | `attention_naive` |
| `model.masking.attention_heads` | `all` |
| `model.masking.attention_layers` | e.g. `last_3` (list in YAML) |

**Config file:** `configs/experiments/C3_naive_attention.yaml`

---

## 2. Implementation notes

1. Run backbone forward with `output_attentions=True` (or hooks), extract **text→image** attention block (PaliGemma may use merged self-attention; slice language rows × image columns — see `experiment/03_sample_data_testing.md` T3).
2. **Aggregate:** mean over **all** heads, mean over language tokens, optionally mean over last 3 layers.
3. **Top-k:** mask the `k = 49` patches with highest saliency (same `mask_ratio` as C2 for fair comparison).

**Snippet (conceptual)**

```python
saliency = attn.mean(dim=1).mean(dim=1)  # (B, P) — adjust dims to your tensor layout
_, topk = saliency.topk(k=49, dim=-1)
mask = torch.zeros(B, P, device=device).scatter_(1, topk, 1.0)
```

---

## 3. What to monitor

| Observation | Meaning |
|-------------|---------|
| `val/action_loss` < C2 | Naive attention masks add useful supervision |
| `val/action_loss` ≈ C2 | Attention saliency ~ random after averaging |
| `val/action_loss` > C2 | **Harmful** masks (attention sinks / noise) — expected risk per `feedback.md` |
| `metrics/aos` rising | Attention focusing on objects |
| `metrics/aid` low | Same saliency for different instructions → degenerate loop risk |

**Extra logs (recommended)**

- `mask/entropy_mean` — dispersion of mask distribution across batch
- `attention/saliency_max` vs `saliency_mean` — sink detection

**Alignment**

- C3 vs C2 is the **first direct test** of “language-guided masking beats random.”
- If C3 loses to C2, proceed to C4 (head selection) before concluding the idea fails.

---

## 4. Run

```bash
python -m training.run_experiment --config configs/experiments/C3_naive_attention.yaml
python scripts/read_tracking.py --expect configs/expectations.yaml --condition C3
```
