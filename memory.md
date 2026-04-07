# Project progress

## Train log / tqdm throttle (~100 per epoch) (2026-04)

- **Goal**: Cut terminal (and default W&B step) spam during long epochs.
- **Feature**: `training.max_logs_per_train_epoch` (default 100): when `len(train_loader)` is known, tqdm `miniters`, postfix, and W&B train scalars align to ~one update per 1% of the epoch (plus last batch). Unknown-length loaders fall back to `training.log_every_n_steps`. Set `max_logs_per_train_epoch: 0` to use only `log_every_n_steps`. If `logging.wandb.log_train_every_n_steps` is set, W&B uses that step interval while tqdm stays throttled. Val tqdm uses the same cap for refresh rate. First-batch batch contract log is DEBUG.
- **Verification**: Run `pytest tests/test_trainer.py` locally (agent env: torch DLL load failure on Windows).

## Continuous 7-DoF action head (2026-04)

- **Goal**: Drop per-DoF binning; train with MSE on continuous actions like the dataset.
- **Architecture**: `ActionHead` outputs `[B, 7]`; `action_loss` = `F.mse_loss`; `dataset_libero` yields float `[B, 7]` targets (no `discretize_action`). Validation logs `val/action_mae_dim_*` and `val/action_mae_mean` via `eval_step_with_action_mae`; W&B keys renamed from `val/action_acc_*`.
- **What I Have Done**: Updated `model.py`, `losses.py`, `metrics.py` (`action_mae_per_dim`), `train.py`, `wandb_training.py`, `tests/*`.
- **Verification**: Run `pytest tests/ -m "not integration"` locally (agent env: torch DLL access violation on Windows).

## Minimal LIBERO dataloader (2026-04)

- **Feature**: `code_base/dataset_libero.py` rewritten: fixed hub `lerobot/libero_spatial_image`, `LiberoLoaderConfig` (batch/loader flags only), `libero_collate_fn` for 112+112 fusion + `task`/`episode_index`/`task_index` prompt + `action` `[7]`; map=`LeRobotDataset`, stream=`load_dataset(..., streaming=True)`; `build_libero_dataloader`, `libero_train_iterator_factory`.
- **Removed**: `LiberoSpatialConfig` options, smoke CLI, task-id filters, separate map/LeRobot dataset wrapper classes (replaced by `LiberoMapRows` / `LiberoStreamRows`).

## Training pipeline wiring and logging (2026-04)

- **Goal**: Confirm train / model / loss / dataset / checkpoint tensor contracts; add step-by-step logging for debugging.
- **Feature**: `validate_training_batch` and `validate_action_logits_shape` in `code_base/train.py`; strict `action_logits` / `action_targets` checks in `compute_batch_losses`; INFO logs in `checkpoint.py`, `dataset_libero.py`, and `LAReconVLATrainer`; `code_base/logging_utils.py` with `configure_training_logging` (root stderr handler), `TRAINING_LOG_LEVEL`, CLI `--log-level`.
- **Architecture**: Data `[B,3,H,W]` + texts + `[B,7]` long bins; model `action_logits` `[B,7,256]`; loss cross-entropy on flattened logits vs targets.
- **What I Have Done**: Implemented the above; tests: `test_validate_training_batch_*`, `test_compute_batch_losses_rejects_logits_shape`; `pytest tests/ -m "not integration"` → 38 passed.
- **Todo List**: Optional: DEBUG logs for first recon tensors when reconstruction is enabled; document `self-learning.md` when established.

## Weights & Biases (2026-04)

- **Goal**: Log experiment-plan metrics and enable API/export for reports and long-run analysis.
- **Feature**: `code_base/wandb_training.py` (`WandbExperimentLogger`, `parse_wandb_settings`, `fetch_run_history`); `logging.wandb` in config + CLI `--no-wandb`; `eval_step_with_accuracy` + val aggregation for `val/action_acc_dim_*`; `scripts/wandb_fetch_run.py`; `code_base/metrics.py` (`action_accuracy_per_dim`).
- **Metrics (align `experiment/02_experiment_plan.md` §5)**: per step — `train/total_loss`, `train/action_loss`, `train/recon_loss`; per epoch — `train/epoch_mean_loss`, `val/total_loss`, `val/action_acc_dim_0..6`, `val/action_acc_mean`, `train/best_metric`, `checkpoint/saved_best`.
- **What I Have Done**: Wired into `LAReconVLATrainer.fit()` with `try/finally` + `wandb.finish()`; updated `plans/PIPELINE.md`; `pandas` in `requirements-training.txt` for fetch script.
- **Verification**: `pytest tests/ -m "not integration"` → 46 passed.

## YAML configs folder (2026-04)

- **Goal**: Versioned experiment configs (C1–C5, ablations, seeds, smoke) aligned with `experiment/02_experiment_plan.md` and `plans/PIPELINE.md`.
- **Feature**: Full standalone `configs/C1.yaml`–`C5.yaml`, `configs/A1_*.yaml`, `configs/A2_*.yaml`; `code_base/config_loader.py`; `train.py` `--config` (multi-file merge supported); exports on `code_base` package.
- **What I Have Done**: Added README under `configs/`; `.gitignore` `checkpoints/`; tests `tests/test_config_loader.py`.
- **Verification**: `pytest tests/ -m "not integration"` → 50 passed.
