# Experiment configs (YAML)

Each file is **self-contained** and **safe to commit** (no API keys). Matches `experiment/02_experiment_plan.md` (C1–C5 + sample ablations).

## Run (repo root)

```bash
python train.py --config configs/C1.yaml
python -m code_base.train --config configs/C2.yaml
```

Disable W&B for a run:

```bash
python train.py --config configs/C1.yaml --no-wandb
```

Merge a second YAML (later file overrides), e.g. gitignored secrets or paths:

```bash
python train.py --config configs/C1.yaml configs/my_machine.local.yaml
```

## Weights & Biases

1. Set **`WANDB_API_KEY`** in the environment (recommended).
2. In YAML: `logging.wandb.enabled: true` and `project: la-reconvla` (change `project` to match your W&B project).
3. Omit `run_name` for an automatic unique name, or set `run_name` for a fixed label.

Do **not** commit `api_key` / `key` in YAML; use `configs/*.local.yaml` (gitignored) if needed.

## Hugging Face / LIBERO

- Optional: **`HF_TOKEN`** for Hub access.
- Optional: **`LIBERO_DATASET_ROOT`** for the local snapshot directory (see `code_base/dataset_libero.py`).
- **`data.libero`**: `batch_size`, `num_workers` (e.g. `2` on Colab), `pin_memory: true` with `training.device: cuda` for faster GPU transfers. `python train.py` builds this loader automatically when `data.libero` is present.

## Files

| File | Description |
|------|-------------|
| `C1.yaml` | Action-only baseline |
| `C2.yaml` | Random-mask MAE, 25%, λ=0.5 |
| `C3.yaml` | Naive attention-mask MAE |
| `C4.yaml` | Selected-head attention MAE — edit `model.masking.selected_heads` after diagnostic |
| `C5.yaml` | Selected-head + EMA teacher masks |
| `A1_lambda_0.1_C2.yaml` | Ablation: λ=0.1 on C2-style MAE |
| `A2_mask_ratio_0.35.yaml` | Ablation: 35% mask on C2-style MAE |

All use `training.use_experiment_preset: false` so behaviour is fully defined in YAML.
