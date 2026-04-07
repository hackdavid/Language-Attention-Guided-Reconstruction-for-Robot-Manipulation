"""
Weights & Biases integration for LA-ReconVLA training.

Logs metrics aligned with ``experiment/02_experiment_plan.md`` §5:
train total / action / recon (per step), val total and per-DoF action accuracy (per epoch).

Also supports fetching runs via the public API (see ``scripts/wandb_fetch_run.py``).
"""

from __future__ import annotations

import copy
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import wandb

from .logging_utils import get_logger

_log = get_logger(__name__)


@dataclass
class WandbSettings:
    enabled: bool = False
    project: str = "la-reconvla"
    # Env WANDB_API_KEY wins; else YAML ``api_key`` or ``key`` (never log the key).
    api_key: Optional[str] = None
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: Sequence[str] = ()
    group: Optional[str] = None
    job_type: str = "train"
    # If set, overrides training.log_every_n_steps for W&B train metrics only.
    log_train_every_n_steps: Optional[int] = None
    # wandb.init(resume=...) e.g. "allow"
    resume: Optional[str] = None
    run_id: Optional[str] = None


def parse_wandb_settings(config_dict: Dict[str, Any]) -> WandbSettings:
    log_cfg = config_dict.get("logging") or {}
    wb = log_cfg.get("wandb") or {}
    if not isinstance(wb, dict):
        return WandbSettings()
    tags = wb.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    lte = wb.get("log_train_every_n_steps")
    if lte is not None:
        lte = max(1, int(lte))
    raw_key = wb.get("api_key") or wb.get("key")
    api_key = str(raw_key).strip() if raw_key not in (None, "") else None
    entity = wb.get("entity")
    if entity is not None and str(entity).strip() == "":
        entity = None
    group = wb.get("group")
    if group is not None and str(group).strip() == "":
        group = None
    run_name = wb.get("run_name")
    if run_name is not None and str(run_name).strip() == "":
        run_name = None
    return WandbSettings(
        enabled=bool(wb.get("enabled", False)),
        project=str(wb.get("project", "la-reconvla")),
        api_key=api_key,
        entity=entity,
        run_name=run_name,
        tags=tuple(str(t) for t in tags),
        group=group,
        job_type=str(wb.get("job_type", "train")),
        log_train_every_n_steps=lte,
        resume=wb.get("resume"),
        run_id=wb.get("run_id") or os.environ.get("WANDB_RUN_ID") or None,
    )


def experiment_run_name(config_dict: Dict[str, Any]) -> Optional[str]:
    exp = config_dict.get("experiment") or {}
    return exp.get("name") or exp.get("condition")


def experiment_tags(config_dict: Dict[str, Any]) -> List[str]:
    exp = config_dict.get("experiment") or {}
    tags: List[str] = []
    for key in ("condition", "name"):
        v = exp.get(key)
        if v is not None and str(v) not in tags:
            tags.append(str(v))
    return tags


def unique_wandb_run_name(config_dict: Dict[str, Any]) -> str:
    """Default run name when YAML ``run_name`` is omitted: ``<experiment>_<UTC_ts>_<short_uuid>``."""
    base = experiment_run_name(config_dict) or "train"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(base))[:64]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{safe}_{ts}_{uuid.uuid4().hex[:8]}"


class WandbExperimentLogger:
    """Thin wrapper: no-op when W&B is disabled in config."""

    def __init__(self, settings: WandbSettings, config_dict: Dict[str, Any]) -> None:
        self.settings = settings
        self._config_dict = copy.deepcopy(config_dict)
        self._run = None

    @property
    def active(self) -> bool:
        return self._run is not None

    def start(self) -> None:
        if not self.settings.enabled:
            return

        env_key = (os.environ.get("WANDB_API_KEY") or "").strip()
        cfg_key = (self.settings.api_key or "").strip()
        key = env_key or cfg_key
        if key:
            try:
                wandb.login(key=key, relogin=True)
                _log.info("W&B: authenticated (API key from %s)", "environment" if env_key else "config")
            except Exception as e:
                _log.warning("W&B login failed (continuing; init may still use cached creds): %s", e)

        if self.settings.run_name:
            name = str(self.settings.run_name).strip()
        else:
            name = unique_wandb_run_name(self._config_dict)

        tags = list(self.settings.tags) + experiment_tags(self._config_dict)
        seen = set()
        tags = [t for t in tags if not (t in seen or seen.add(t))]

        kwargs: Dict[str, Any] = {
            "project": self.settings.project,
            "name": name,
            "config": self._config_dict,
            "tags": tags or None,
            "job_type": self.settings.job_type,
        }
        if self.settings.entity:
            kwargs["entity"] = self.settings.entity
        if self.settings.group:
            kwargs["group"] = self.settings.group
        if self.settings.run_id:
            kwargs["id"] = self.settings.run_id
        if self.settings.resume:
            kwargs["resume"] = self.settings.resume

        try:
            self._run = wandb.init(**kwargs)
        except Exception as e:
            _log.warning("W&B init failed (continuing without W&B): %s", e)
            self._run = None
            return
        _log.info(
            "W&B run started: project=%s name=%s id=%s",
            self.settings.project,
            name,
            getattr(self._run, "id", None),
        )

    def log_train_step(self, global_step: int, loss_dict: Dict[str, float]) -> None:
        if self._run is None:
            return

        payload = {
            "train/total_loss": loss_dict["total"],
            "train/action_loss": loss_dict["action"],
            "train/recon_loss": loss_dict["recon"],
        }
        wandb.log(payload, step=global_step)

    def log_epoch_end(
        self,
        global_step: int,
        *,
        epoch_1based: int,
        mean_train_loss: float,
        mean_val_loss: Optional[float],
        val_action_mae_per_dim: Optional[Sequence[float]],
        best_metric: float,
        checkpoint_saved_best: bool,
    ) -> None:
        if self._run is None:
            return

        payload: Dict[str, Any] = {
            "epoch": epoch_1based,
            "train/epoch_mean_loss": mean_train_loss,
            "train/best_metric": best_metric,
            "checkpoint/saved_best": float(checkpoint_saved_best),
        }
        if mean_val_loss is not None:
            payload["val/total_loss"] = mean_val_loss
        if val_action_mae_per_dim is not None and len(val_action_mae_per_dim) > 0:
            for i, mae in enumerate(val_action_mae_per_dim):
                payload[f"val/action_mae_dim_{i}"] = mae
            payload["val/action_mae_mean"] = float(sum(val_action_mae_per_dim) / len(val_action_mae_per_dim))
        wandb.log(payload, step=global_step)

    def finish(self) -> None:
        if self._run is None:
            return

        wandb.finish()
        self._run = None
        _log.info("W&B run finished")


def fetch_run_history(
    path: str,
    *,
    keys: Optional[Sequence[str]] = None,
    pandas: bool = True,
):
    """
    Public API: load scalar history for analysis (scripts / notebooks).

    Args:
        path: ``entity/project/run_id`` or full run path as in the W&B UI.
        keys: Optional metric names to restrict columns.
        pandas: If True (default), return a pandas DataFrame; else list of dict rows.

    Returns:
        DataFrame or list of step rows.
    """
    api = wandb.Api()
    run = api.run(path)
    hist = run.scan_history(keys=keys)
    if pandas:
        return pd.DataFrame(hist)
    return list(hist)
