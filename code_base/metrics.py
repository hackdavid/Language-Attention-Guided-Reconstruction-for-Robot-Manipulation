"""Batch metrics for experiment logging (e.g. per-DoF action MAE)."""

from __future__ import annotations

import torch


@torch.no_grad()
def action_mae_per_dim(action_pred: torch.Tensor, action_targets: torch.Tensor) -> torch.Tensor:
    """
    Per-dimension mean absolute error over the batch.

    Args:
        action_pred: [B, 7]
        action_targets: [B, 7]

    Returns:
        Tensor of shape [7] with MAE per DoF.
    """
    if action_pred.dim() != 2 or action_targets.dim() != 2:
        raise ValueError(
            f"Expected pred and targets [B,7]; got {tuple(action_pred.shape)}, {tuple(action_targets.shape)}"
        )
    if action_pred.shape != action_targets.shape:
        raise ValueError(
            f"Shape mismatch: pred {tuple(action_pred.shape)} vs targets {tuple(action_targets.shape)}"
        )
    return (action_pred.float() - action_targets.float()).abs().mean(dim=0)
