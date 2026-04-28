# Config-driven experiment pipeline

## Layout

| Path | Role |
|------|------|
| `configs/C1.yaml` … `configs/C5.yaml` | Full self-contained experiment configs |
| `configs/A1_*.yaml`, `configs/A2_*.yaml` | Full ablation configs |
| `configs/expectations.yaml` | Gates for `scripts/read_tracking.py` (smoke + condition checklists) |
| `training/config_loader.py` | Deep-merge `base` + experiment YAML |
| `training/trackers.py` | **MLflow** (local `file:./mlruns`) + **W&B** (optional) |
| `training/run_experiment.py` | CLI entry; `--smoke` verifies logging without a model |
| `scripts/read_tracking.py` | Latest run + expectation check (exit 0/1 for CI) |
| `scripts/download_data_hf.py` | HuggingFace snapshot / list dataset |
| `scripts/summarize_wandb_offline.py` | Print latest offline W&B summary JSON |

## Switch experiment

Single full config (or merge overrides):

```bash
python -m code_base.train --config configs/C4.yaml
```

Edit `model.masking.selected_heads` in `configs/C4.yaml` / `configs/C5.yaml` after the Step 1 diagnostic. Optional: `python -m training.run_experiment --config ...` if that entrypoint is wired.

## MLflow (local)

- URI: `file:./mlruns` (set in your merged YAML if using MLflow)
- View UI: `mlflow ui --backend-store-uri file:./mlruns`

## Weights & Biases

1. `wandb login` (once).
2. In the merged training config, set `logging.wandb.enabled: true` (see `code_base.train.default_train_config_dict()` for keys: `project`, `run_name`, `tags`, `group`, `run_id`, `resume`).
3. Training logs metrics from `experiment/02_experiment_plan.md` §5: `train/total_loss`, `train/action_loss`, `train/recon_loss` (per step); `val/total_loss`, `val/action_acc_dim_0..6`, `val/action_acc_mean` (per epoch when `training.val_batches` > 0).
4. For air-gapped runs: `set WANDB_MODE=offline` (Windows) then later `wandb sync wandb/offline-run-*`.
5. Fetch history for reports: `python scripts/wandb_fetch_run.py ENTITY/PROJECT/RUN_ID --out metrics.csv` (requires network + API key).

**CLI:** `python -m code_base.train --no-wandb` forces W&B off (CI / smoke). The older `training/run_experiment.py` `--smoke` pattern applies if you use that entrypoint.

## CI alignment check

After a run:

```bash
python scripts/read_tracking.py --expect configs/expectations.yaml --condition C4
```

- Exit code **0**: smoke expectations passed (or no expect file).
- Exit code **1**: at least one check failed (e.g. loss did not decrease).

Tune `configs/expectations.yaml` → `real:` section once you have real baselines.

## Next step (real training)

Replace `NotImplementedError` in `training/run_experiment.py` → `run_real_placeholder` by calling your `train.py` loop from `doc/PART2_IMPLEMENTATION_GUIDE.md`, passing the merged `cfg` dict and using `ExperimentTrackers.log_metrics` each step/epoch.
