# LA-ReconVLA — Research plans index

This folder ties together **data**, **per-condition experiments**, and the **config-driven pipeline** under `configs/` and `training/`.

| Document | Purpose |
|----------|---------|
| [01_dataset_libero_and_reconvla.md](01_dataset_libero_and_reconvla.md) | LIBERO / RLDS / HuggingFace: what to use, sizes, streaming vs local, how ReconVLA used LIBERO |
| [exp_C1_baseline.md](exp_C1_baseline.md) | Condition C1: action-only baseline |
| [exp_C2_random_mae.md](exp_C2_random_mae.md) | Condition C2: random-mask MAE |
| [exp_C3_naive_attention.md](exp_C3_naive_attention.md) | Condition C3: naive attention masking |
| [exp_C4_selected_heads.md](exp_C4_selected_heads.md) | Condition C4: selected localization heads |
| [exp_C5_ema_teacher.md](exp_C5_ema_teacher.md) | Condition C5: EMA teacher masks |
| [exp_ablations.md](exp_ablations.md) | A1–A3 (λ, mask ratio, contiguous masks) |
| [PIPELINE.md](PIPELINE.md) | YAML merge, MLflow + W&B, CI checks, wiring real training |

**Pipeline entry**

```bash
# Install deps (see project requirements)
pip install pyyaml wandb mlflow

# Smoke test: synthetic loop, logs to MLflow (./mlruns) + W&B (offline if no key)
python -m training.run_experiment --config configs/experiments/C1_action_only.yaml --smoke

# Read last runs and check vs expectations (for humans + CI)
python scripts/read_tracking.py --mlflow-uri file:./mlruns --expect configs/expectations.yaml
python scripts/summarize_wandb_offline.py
```

**Config layout**

- `configs/base.yaml` — shared defaults (data paths, model, training, logging).
- `configs/experiments/*.yaml` — one file per condition; only overrides + `experiment.name`.

Change experiment by swapping `--config` (no code edits required for condition switches).
