"""
Checkpoint I/O: latest (resume) and best (metric-driven) weights + optimizer state.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .logging_utils import get_logger

_log = get_logger(__name__)

CHECKPOINT_FORMAT_VERSION = 1
LATEST_FILENAME = "latest.pt"
BEST_FILENAME = "best.pt"


def collect_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = None
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore Python / NumPy / PyTorch RNG state from a checkpoint payload."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def build_checkpoint_payload(
    *,
    next_epoch: int,
    global_step: int,
    best_metric: float,
    model_state_dict: Dict[str, torch.Tensor],
    optimizer_state_dict: Dict[str, Any],
    scaler_state_dict: Optional[Dict[str, Any]],
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "next_epoch": next_epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scaler_state_dict": scaler_state_dict,
        "rng_state": collect_rng_state(),
        "config_dict": config_dict,
    }


def load_checkpoint(path: Path, map_location: Optional[Any] = None) -> Dict[str, Any]:
    """Load a checkpoint dict saved by :class:`CheckpointManager`."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint (not a dict): {path}")
    ver = ckpt.get("format_version", 0)
    if ver != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format_version={ver!r} (expected {CHECKPOINT_FORMAT_VERSION})"
        )
    return ckpt


@dataclass
class ResumeState:
    next_epoch: int
    global_step: int
    best_metric: float


class CheckpointManager:
    """Writes ``latest.pt`` and ``best.pt`` under a run directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.directory / LATEST_FILENAME
        self.best_path = self.directory / BEST_FILENAME

    def save_latest(self, payload: Dict[str, Any]) -> None:
        torch.save(payload, self.latest_path)
        _log.info("Saved checkpoint: %s (next_epoch=%s global_step=%s)", self.latest_path, payload.get("next_epoch"), payload.get("global_step"))

    def save_best(self, payload: Dict[str, Any]) -> None:
        torch.save(payload, self.best_path)
        _log.info("Saved best checkpoint: %s (best_metric=%s)", self.best_path, payload.get("best_metric"))


def load_training_state(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any] = None,
    map_location: Optional[Any] = None,
) -> ResumeState:
    """Load model/optimizer/scaler and RNG; return resume cursor."""
    _log.info("Loading training state from %s", checkpoint_path.resolve())
    ckpt = load_checkpoint(checkpoint_path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    s_state = ckpt.get("scaler_state_dict")
    if scaler is not None and s_state is not None:
        scaler.load_state_dict(s_state)
    rng = ckpt.get("rng_state")
    if rng is not None:
        restore_rng_state(rng)
    state = ResumeState(
        next_epoch=int(ckpt["next_epoch"]),
        global_step=int(ckpt["global_step"]),
        best_metric=float(ckpt["best_metric"]),
    )
    _log.info(
        "Resume cursor: next_epoch=%s global_step=%s best_metric=%s",
        state.next_epoch,
        state.global_step,
        state.best_metric,
    )
    return state
