"""
Training loop for LAReconVLA: losses in ``losses.py``, checkpoints in ``checkpoint.py``.

Run from the repository root::

    python -m code_base.train
    python -m code_base.train --config configs/C1.yaml
    python -m code_base.train --config path/to/config.yaml --resume path/to/latest.pt
    python -m code_base.train --no-wandb --log-level INFO

If the merged config includes ``data.libero``, training uses the LIBERO DataLoader; otherwise
dummy in-memory batches (``training.batch_size``) are used. Set ``training.device: cuda`` for GPU;
the trainer raises a clear error if CUDA is requested but unavailable.

Enable W&B in YAML under ``logging.wandb.enabled: true``; set ``WANDB_API_KEY`` (or optional YAML ``api_key`` / ``key``). Omit ``run_name`` for an automatic unique name. Offline: ``WANDB_MODE=offline`` then ``wandb sync``.
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .checkpoint import (
    CheckpointManager,
    LATEST_FILENAME,
    build_checkpoint_payload,
    load_training_state,
)
from .logging_utils import configure_training_logging, get_logger, log_level_from_env
from .losses import compute_batch_losses, detach_loss_dict
from .model import ActionHead, LAReconVLA, LAReconVLAConfigSource, ModelConfig
from .config_loader import load_merged_config, load_one_config_file
from .wandb_training import WandbExperimentLogger, parse_wandb_settings


def default_train_config_dict() -> Dict[str, Any]:
    return {
        "experiment": {"condition": "C1"},
        "model": {
            "freeze_backbone": True,
            "finetune_last_n_layers": 0,
            "backbone": {
                "model_id": "google/paligemma-3b-pt-224",
                "torch_dtype": "bfloat16",
                "device_map": None,
            },
            "reconstruction": {"enabled": False, "lambda_recon": 0.0},
            "masking": {"mode": "none"},
        },
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "batches_per_epoch": 2,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "device": "cpu",
            "seed": 42,
            "use_experiment_preset": False,
            "mixed_precision": False,
            "log_every_n_steps": 1,
            # When >0 and epoch length is known, tqdm/W&B train logs ~this many times per epoch
            # (every ~1% if 100). When 0, use log_every_n_steps only. Unknown len → log_every_n_steps.
            "max_logs_per_train_epoch": 100,
            "checkpoint_dir": None,
            "val_batches": 0,
            "resume_from": None,
            "best_checkpoint_metric": "train_loss",
        },
        "logging": {
            "wandb": {
                "enabled": False,
                "project": "la-reconvla",
                "entity": None,
                "run_name": None,
                "tags": [],
                "group": None,
                "job_type": "train",
                "log_train_every_n_steps": None,
                "resume": None,
                "run_id": None,
            }
        },
    }


def load_config_file(path: Path) -> Dict[str, Any]:
    """Load a single YAML/JSON config file."""
    return load_one_config_file(path)


def validate_training_batch(
    images: torch.Tensor,
    texts: Sequence[str],
    action_targets: torch.Tensor,
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Assert trainer data contract: images [B,3,H,W], texts length B, action_targets [B,7] float."""
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(f"images must be [B, 3, H, W], got shape {tuple(images.shape)}")
    b = images.shape[0]
    if len(texts) != b:
        raise ValueError(f"len(texts)={len(texts)} != batch size B={b}")
    if tuple(action_targets.shape) != (b, ActionHead.NUM_ACTION_DIMS):
        raise ValueError(
            f"action_targets must be [B, {ActionHead.NUM_ACTION_DIMS}], got {tuple(action_targets.shape)}"
        )
    if action_targets.dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
        raise ValueError(
            f"action_targets must be a float dtype (e.g. float32), got {action_targets.dtype}"
        )
    if logger is not None:
        _, _, h, w = images.shape
        logger.debug(
            "Batch contract: B=%s images=[B,3,%s,%s] action_targets=[%s,%s] dtype=%s",
            b,
            h,
            w,
            b,
            ActionHead.NUM_ACTION_DIMS,
            action_targets.dtype,
        )


def validate_action_logits_shape(logits: torch.Tensor, batch_size: int) -> None:
    exp = (batch_size, ActionHead.NUM_ACTION_DIMS)
    if tuple(logits.shape) != exp:
        raise ValueError(f"action_logits must be {exp}, got {tuple(logits.shape)}")


@dataclass
class TrainingSettings:
    epochs: int = 1
    batch_size: int = 1
    batches_per_epoch: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    device: str = "cpu"
    seed: int = 42
    use_experiment_preset: bool = False
    mixed_precision: bool = False
    log_every_n_steps: int = 1
    max_logs_per_train_epoch: int = 100
    checkpoint_dir: Optional[str] = None
    val_batches: int = 0
    resume_from: Optional[str] = None
    best_checkpoint_metric: str = "train_loss"


class LAReconVLATrainer:
    """
    Builds ``LAReconVLA`` from ``config_dict``, optimizes using losses from ``compute_batch_losses``.

    Data: ``fit(train_loader=...)`` expects a factory returning iterators of
    ``(images, texts, action_targets)`` with ``images`` [B,3,H,W], ``action_targets`` [B,7] float32.

    LIBERO: with ``data.libero`` in the YAML, :func:`train_loader_factory_from_config` (used by
    ``main()``) builds :func:`code_base.dataset_libero.libero_train_iterator_factory` automatically.
    Or pass ``train_loader`` explicitly when constructing ``fit()`` from a notebook.

    Checkpoints: set ``training.checkpoint_dir`` to enable ``latest.pt`` each epoch and
    ``best.pt`` when the monitored metric improves. Use ``training.resume_from`` as a path
    to a ``.pt`` file or the string ``\"latest\"`` (uses ``checkpoint_dir / latest.pt``).

    Logging: from a script, call ``configure_training_logging()`` (or use ``python -m code_base.train --log-level INFO``)
    so dataset/checkpoint/trainer INFO lines appear on stderr.

    W&B (``experiment/02_experiment_plan.md`` §5): ``logging.wandb.enabled: true``, ``project``, and ``WANDB_API_KEY`` (or YAML ``api_key``). Leave ``run_name`` unset for a generated unique name per run.
    Logs ``train/total_loss``, ``train/action_loss``, ``train/recon_loss`` at a capped rate: by default
    ``training.max_logs_per_train_epoch`` (100) when the loader length is known (~one log per 1% of the epoch),
    else ``training.log_every_n_steps``. If ``logging.wandb.log_train_every_n_steps`` is set, W&B
    logs at that step interval regardless (tqdm stays on the capped cadence). Each epoch end:
    ``train/epoch_mean_loss``, ``val/total_loss``, ``val/action_mae_dim_*``,
    ``val/action_mae_mean``. Fetch runs programmatically via ``code_base.wandb_training.fetch_run_history`` or
    ``python scripts/wandb_fetch_run.py``.
    """

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        self._log = get_logger(__name__)
        self.config_dict = copy.deepcopy(config_dict)
        self.training = self._parse_training(self.config_dict.get("training") or {})
        self._set_seed(self.training.seed)

        self.model_cfg: ModelConfig = LAReconVLAConfigSource(
            self.config_dict,
            use_experiment_preset=self.training.use_experiment_preset,
        ).model_config()
        self.model = LAReconVLA(self.model_cfg)

        dev_str = str(self.training.device).strip()
        if dev_str.lower().startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "training.device requests CUDA but torch.cuda.is_available() is False. "
                "In Colab use Runtime → Change runtime type → GPU (T4), or set training.device: cpu."
            )
        self.device = torch.device(dev_str)
        self._single_device = self.model_cfg.backbone.device_map is None
        if self._single_device:
            self.model.to(self.device)
        else:
            self.device = next(self.model.parameters()).device

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            props = torch.cuda.get_device_properties(self.device)
            self._log.info(
                "CUDA active: %s | %.2f GiB total VRAM",
                torch.cuda.get_device_name(self.device),
                props.total_memory / (1024.0**3),
            )

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters: check freeze_backbone / model config")
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.training.learning_rate,
            weight_decay=self.training.weight_decay,
        )
        use_amp = self.training.mixed_precision and self.training.device.startswith("cuda")
        self._scaler: Optional[GradScaler] = GradScaler() if use_amp else None

        self._global_step = 0
        self._resume_next_epoch = 0
        self._best_metric = float("inf")
        self._batch_contract_checked = False
        self._forward_shapes_logged = False
        self._wandb_settings = parse_wandb_settings(self.config_dict)

        resume_path = self._resolve_resume_path()
        if resume_path is not None:
            self._load_checkpoint(resume_path)

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._log.info(
            "Trainer ready: device=%s trainable_params=%s epochs=%s batch_size=%s "
            "batches_per_epoch=%s checkpoint_dir=%s resume_next_epoch=%s global_step=%s",
            self.device,
            n_trainable,
            self.training.epochs,
            self.training.batch_size,
            self.training.batches_per_epoch,
            self.training.checkpoint_dir,
            self._resume_next_epoch,
            self._global_step,
        )

    def _resolve_resume_path(self) -> Optional[Path]:
        rf = self.training.resume_from
        if not rf:
            return None
        if rf == "latest":
            if not self.training.checkpoint_dir:
                raise ValueError("training.resume_from='latest' requires training.checkpoint_dir")
            return Path(self.training.checkpoint_dir) / LATEST_FILENAME
        return Path(rf).expanduser().resolve()

    def _load_checkpoint(self, path: Path) -> None:
        map_loc = self.device if self._single_device else None
        state = load_training_state(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self._scaler,
            map_location=map_loc,
        )
        self._resume_next_epoch = state.next_epoch
        self._global_step = state.global_step
        self._best_metric = state.best_metric

    @staticmethod
    def _parse_training(d: Dict[str, Any]) -> TrainingSettings:
        return TrainingSettings(
            epochs=max(1, int(d.get("epochs", 1))),
            batch_size=max(1, int(d.get("batch_size", 1))),
            batches_per_epoch=max(1, int(d.get("batches_per_epoch", 4))),
            learning_rate=float(d.get("learning_rate", 1e-5)),
            weight_decay=float(d.get("weight_decay", 0.01)),
            max_grad_norm=float(d.get("max_grad_norm", 1.0)),
            device=str(d.get("device", "cpu")),
            seed=int(d.get("seed", 42)),
            use_experiment_preset=bool(d.get("use_experiment_preset", False)),
            mixed_precision=bool(d.get("mixed_precision", False)),
            log_every_n_steps=max(1, int(d.get("log_every_n_steps", 1))),
            max_logs_per_train_epoch=max(0, int(d.get("max_logs_per_train_epoch", 100))),
            checkpoint_dir=d.get("checkpoint_dir"),
            val_batches=int(d.get("val_batches", 0)),
            resume_from=d.get("resume_from"),
            best_checkpoint_metric=str(d.get("best_checkpoint_metric", "train_loss")),
        )

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _move_batch(
        self, images: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._single_device:
            nb = self.device.type == "cuda" and images.is_pinned()
            return (
                images.to(self.device, non_blocking=nb),
                actions.to(self.device, non_blocking=nb),
            )
        return images, actions

    def _losses_from_outputs(self, outputs: Dict[str, Any], action_targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        return compute_batch_losses(
            outputs,
            action_targets,
            lambda_recon=self.model_cfg.reconstruction.lambda_recon,
            reconstruction_enabled=self.model_cfg.reconstruction.enabled,
        )

    def train_step(self, images: torch.Tensor, texts: Sequence[str], action_targets: torch.Tensor) -> Dict[str, float]:
        if not self._batch_contract_checked:
            validate_training_batch(images, texts, action_targets, logger=self._log)
            self._batch_contract_checked = True

        self.model.train()
        images, action_targets = self._move_batch(images, action_targets)
        self.optimizer.zero_grad(set_to_none=True)

        if self._scaler is not None:
            with autocast():
                outputs = self.model(images, texts)
                losses = self._losses_from_outputs(outputs, action_targets)
                loss = losses["total"]
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training.max_grad_norm)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            outputs = self.model(images, texts)
            losses = self._losses_from_outputs(outputs, action_targets)
            loss = losses["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training.max_grad_norm)
            self.optimizer.step()

        if not self._forward_shapes_logged:
            validate_action_logits_shape(outputs["action_logits"], images.shape[0])
            self._log.info("Forward contract: action_logits=%s", tuple(outputs["action_logits"].shape))
            self._forward_shapes_logged = True

        if self.model_cfg.ema.enabled:
            self.model.update_ema_teacher()

        self._global_step += 1
        return detach_loss_dict(losses)

    @torch.no_grad()
    def eval_step(self, images: torch.Tensor, texts: Sequence[str], action_targets: torch.Tensor) -> Dict[str, float]:
        self.model.eval()
        images, action_targets = self._move_batch(images, action_targets)
        if self._scaler is not None:
            with autocast():
                outputs = self.model(images, texts)
                losses = self._losses_from_outputs(outputs, action_targets)
        else:
            outputs = self.model(images, texts)
            losses = self._losses_from_outputs(outputs, action_targets)
        return detach_loss_dict(losses)

    @torch.no_grad()
    def eval_step_with_action_mae(
        self, images: torch.Tensor, texts: Sequence[str], action_targets: torch.Tensor
    ) -> Tuple[Dict[str, float], torch.Tensor, int]:
        """
        Returns detached loss dict, sum of absolute errors per DoF [7] over the batch, and batch size
        (for aggregating validation MAE: divide cumulative sums by total sample count).
        """
        self.model.eval()
        images, action_targets = self._move_batch(images, action_targets)
        b = images.shape[0]
        if self._scaler is not None:
            with autocast():
                outputs = self.model(images, texts)
                losses = self._losses_from_outputs(outputs, action_targets)
        else:
            outputs = self.model(images, texts)
            losses = self._losses_from_outputs(outputs, action_targets)
        pred = outputs["action_logits"].float()
        tgt = action_targets.float()
        sum_abs_per_dim = (pred - tgt).abs().sum(dim=0)
        return detach_loss_dict(losses), sum_abs_per_dim, b

    def _dummy_batch(self) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        b = self.training.batch_size
        size = 224
        images = torch.rand(b, 3, size, size)
        texts = ["answer en Pick up the object ."] * b
        actions = torch.randn(b, 7, dtype=torch.float32)
        return images, texts, actions

    def _iter_dummy_batches(self) -> Iterator[Tuple[torch.Tensor, List[str], torch.Tensor]]:
        for _ in range(self.training.batches_per_epoch):
            yield self._dummy_batch()

    def _metric_for_best(self, mean_train: float, mean_val: Optional[float]) -> float:
        want = self.training.best_checkpoint_metric
        if want == "val_loss" and mean_val is not None:
            return mean_val
        if want == "val_loss" and mean_val is None and self.training.val_batches <= 0:
            warnings.warn(
                "best_checkpoint_metric is val_loss but no validation is configured; using train_loss.",
                UserWarning,
                stacklevel=2,
            )
        return mean_train

    def _build_ckpt_payload(self, next_epoch: int) -> Dict[str, Any]:
        return build_checkpoint_payload(
            next_epoch=next_epoch,
            global_step=self._global_step,
            best_metric=self._best_metric,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scaler_state_dict=self._scaler.state_dict() if self._scaler is not None else None,
            config_dict=self.config_dict,
        )

    def fit(
        self,
        train_loader: Optional[Callable[[], Iterator[Tuple[torch.Tensor, Sequence[str], torch.Tensor]]]] = None,
        val_loader: Optional[Callable[[], Iterator[Tuple[torch.Tensor, Sequence[str], torch.Tensor]]]] = None,
    ) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        start = self._resume_next_epoch
        if start >= self.training.epochs:
            warnings.warn(
                f"next_epoch={start} >= training.epochs={self.training.epochs}; nothing to run.",
                UserWarning,
                stacklevel=2,
            )
            return history

        wb = WandbExperimentLogger(self._wandb_settings, self.config_dict)
        wb.start()
        wb_train_step_every = self._wandb_settings.log_train_every_n_steps

        ckpt_mgr: Optional[CheckpointManager] = None
        if self.training.checkpoint_dir:
            ckpt_mgr = CheckpointManager(Path(self.training.checkpoint_dir))
            self._log.info("Checkpoints: directory=%s", self.training.checkpoint_dir)

        if train_loader is None:
            self._log.info(
                "fit(): using dummy data (batches_per_epoch=%s, batch_size=%s)",
                self.training.batches_per_epoch,
                self.training.batch_size,
            )
        else:
            self._log.info("fit(): using custom train_loader factory")

        try:
            epoch_pbar = tqdm(
                range(start, self.training.epochs),
                desc="Epochs",
                position=0,
                leave=True,
                initial=start,
                total=self.training.epochs,
            )
            for epoch in epoch_pbar:
                self._log.info("Epoch %s/%s starting", epoch + 1, self.training.epochs)
                loader = train_loader() if train_loader is not None else self._iter_dummy_batches()
                try:
                    epoch_num_batches = len(loader)
                except TypeError:
                    epoch_num_batches = None
                ml = self.training.max_logs_per_train_epoch
                if epoch_num_batches is not None and ml > 0:
                    train_log_interval = max(1, int(epoch_num_batches) // ml)
                else:
                    train_log_interval = max(1, int(self.training.log_every_n_steps))

                step_losses: List[float] = []
                batch_pbar = tqdm(
                    loader,
                    desc=f"Train ep {epoch + 1}/{self.training.epochs}",
                    position=1,
                    leave=False,
                    total=(
                        self.training.batches_per_epoch
                        if train_loader is None
                        else epoch_num_batches
                    ),
                    miniters=train_log_interval,
                    mininterval=0.25,
                )
                for batch_idx, batch in enumerate(batch_pbar):
                    images, texts, actions = batch
                    loss_dict = self.train_step(images, texts, actions)
                    total = loss_dict["total"]
                    step_losses.append(total)
                    n_done = batch_idx + 1
                    is_last = epoch_num_batches is not None and n_done == epoch_num_batches
                    should_log = (n_done % train_log_interval == 0) or is_last
                    if should_log:
                        batch_pbar.set_postfix(loss=f"{total:.4f}", action=f"{loss_dict['action']:.4f}")
                    if wb.active:
                        wb_dense = (
                            wb_train_step_every is not None
                            and self._global_step % max(1, int(wb_train_step_every)) == 0
                        )
                        if wb_dense or (
                            wb_train_step_every is None and should_log
                        ):
                            wb.log_train_step(self._global_step, loss_dict)

                mean_train = float(np.mean(step_losses)) if step_losses else 0.0
                history["train_loss"].append(mean_train)
                self._log.info(
                    "Epoch %s/%s train_loss=%.6f steps=%s",
                    epoch + 1,
                    self.training.epochs,
                    mean_train,
                    len(step_losses),
                )
                epoch_pbar.set_postfix(train_loss=f"{mean_train:.4f}")

                mean_val: Optional[float] = None
                val_mae_per_dim: Optional[List[float]] = None
                if self.training.val_batches > 0:
                    v_loader = val_loader() if val_loader is not None else self._iter_dummy_batches()
                    v_losses: List[float] = []
                    sum_abs = torch.zeros(ActionHead.NUM_ACTION_DIMS, dtype=torch.float64)
                    sum_n = 0
                    vb = self.training.val_batches
                    v_ml = self.training.max_logs_per_train_epoch
                    val_miniters = max(1, vb // v_ml) if v_ml > 0 else 1
                    for i, batch in enumerate(
                        tqdm(
                            v_loader,
                            desc="Val",
                            position=1,
                            leave=False,
                            total=vb,
                            miniters=val_miniters,
                            mininterval=0.25,
                        )
                    ):
                        if i >= self.training.val_batches:
                            break
                        images, texts, actions = batch
                        v_ld, abs_d, bsz = self.eval_step_with_action_mae(images, texts, actions)
                        v_losses.append(v_ld["total"])
                        sum_abs += abs_d.double().cpu()
                        sum_n += bsz
                    mean_val = float(np.mean(v_losses)) if v_losses else 0.0
                    if sum_n > 0:
                        val_mae_per_dim = (sum_abs / float(sum_n)).tolist()
                    history["val_loss"].append(mean_val)
                    epoch_pbar.set_postfix(train_loss=f"{mean_train:.4f}", val_loss=f"{mean_val:.4f}")

                monitored = self._metric_for_best(mean_train, mean_val)
                next_epoch = epoch + 1
                saved_best = False

                if ckpt_mgr is not None:
                    improved = monitored < self._best_metric
                    if improved:
                        self._best_metric = monitored
                    payload = self._build_ckpt_payload(next_epoch=next_epoch)
                    ckpt_mgr.save_latest(payload)
                    if improved:
                        ckpt_mgr.save_best(payload)
                        saved_best = True
                        epoch_pbar.set_postfix(
                            train_loss=f"{mean_train:.4f}",
                            val_loss=f"{mean_val:.4f}" if mean_val is not None else "",
                            best="saved",
                        )

                wb.log_epoch_end(
                    self._global_step,
                    epoch_1based=epoch + 1,
                    mean_train_loss=mean_train,
                    mean_val_loss=mean_val,
                    val_action_mae_per_dim=val_mae_per_dim,
                    best_metric=self._best_metric,
                    checkpoint_saved_best=saved_best,
                )
        finally:
            wb.finish()

        return history


def train_loader_factory_from_config(
    config_dict: Dict[str, Any],
) -> Optional[Callable[[], Iterator[Tuple[torch.Tensor, Sequence[str], torch.Tensor]]]]:
    """
    If ``data.libero`` is present, build :func:`code_base.dataset_libero.libero_train_iterator_factory`.

    When ``training.device`` is CUDA, defaults ``pin_memory`` to True so batches can use
    ``non_blocking`` host-to-device copies.
    """
    lib = (config_dict.get("data") or {}).get("libero")
    if not isinstance(lib, dict):
        return None
    from .dataset_libero import LiberoLoaderConfig, libero_train_iterator_factory

    tr = config_dict.get("training") or {}
    want_cuda = str(tr.get("device", "cpu")).lower().startswith("cuda")
    pin_default = bool(want_cuda and torch.cuda.is_available())
    seed = lib.get("seed")
    if seed is not None:
        seed = int(seed)
    lc = LiberoLoaderConfig(
        batch_size=max(1, int(lib.get("batch_size", 4))),
        num_workers=max(0, int(lib.get("num_workers", 0))),
        shuffle=bool(lib.get("shuffle", True)),
        seed=seed,
        local_root=lib.get("local_root"),
        skip_snapshot_download=bool(lib.get("skip_snapshot_download", False)),
        pin_memory=bool(lib.get("pin_memory", pin_default)),
    )
    log = get_logger(__name__)
    log.info(
        "LIBERO train loader: batch_size=%s num_workers=%s pin_memory=%s",
        lc.batch_size,
        lc.num_workers,
        lc.pin_memory,
    )
    return libero_train_iterator_factory(lc)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train LAReconVLA from a config dict (YAML/JSON).")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        default=None,
        help="One or more .yaml/.json paths, deep-merged left to right (e.g. configs/C1.yaml my_override.yaml).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to resume, or 'latest' with training.checkpoint_dir set in config.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Stderr log level (default: TRAINING_LOG_LEVEL or INFO).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Force-disable W&B even if logging.wandb.enabled is true in config (CI / smoke).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.log_level is not None:
        configure_training_logging(getattr(logging, args.log_level))
    else:
        configure_training_logging(log_level_from_env(logging.INFO))

    if args.config:
        if len(args.config) == 1:
            cfg = load_config_file(Path(args.config[0]))
        else:
            cfg = load_merged_config([Path(p) for p in args.config])
    else:
        cfg = default_train_config_dict()

    if args.resume:
        cfg.setdefault("training", {})
        cfg["training"]["resume_from"] = args.resume

    if args.no_wandb:
        cfg.setdefault("logging", {})
        cfg["logging"].setdefault("wandb", {})
        cfg["logging"]["wandb"]["enabled"] = False

    log = get_logger(__name__)
    log.info(
        "Starting LAReconVLA training (config=%s)",
        args.config if args.config else "defaults",
    )
    loader_factory = train_loader_factory_from_config(cfg)
    if loader_factory is None:
        log.info("No data.libero in config: using in-memory dummy batches (training.batch_size).")
    trainer = LAReconVLATrainer(cfg)
    trainer.fit(train_loader=loader_factory)


if __name__ == "__main__":
    main()
