# Ablations A1–A3

Run **after** identifying the best core condition (usually C4 or C5).

---

## A1 — λ sweep (`lambda_recon`)

**Variables:** `0.1`, `0.5`, `1.0`  
**Config pattern:** `configs/experiments/ablation_A1_lambda_0.1.yaml`, etc. (inherit from best condition, override only `model.reconstruction.lambda_recon`).

**Monitor**

- `val/action_loss` vs `val/recon_loss` trade-off
- If λ too high: action metric degrades

---

## A2 — Masking ratio

**Variables:** `0.15`, `0.25`, `0.35` → `k = floor(ratio * 196)`  
**Override:** `model.masking.mask_ratio`

**Monitor**

- `val/recon_mse` vs action metrics (harder mask → harder recon)
- Too easy recon (loss → 0 fast) at 0.15

---

## A3 — Contiguous vs scattered mask

**Variables:** `model.masking.topology: scattered | contiguous`  
**Contiguous:** take top-*k* seeds then grow connected region on 14×14 grid to same area.

**Monitor**

- AOS and TSR on relational task T1 vs T2–T3

---

## Run batch example

```bash
for y in configs/experiments/ablation_A1_lambda_*.yaml; do
  python -m training.run_experiment --config "$y"
done
```

Use `scripts/read_tracking.py` with `--tag ablation_A1` if you log a run tag in YAML.
