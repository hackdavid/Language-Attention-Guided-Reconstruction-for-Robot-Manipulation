"""
Training-time loss computation for LAReconVLA.

The model returns logits and reconstruction tensors only; this module combines them
with targets and config into scalar losses for backprop.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from .model import ActionHead


def action_mse_loss(action_pred: torch.Tensor, action_targets: torch.Tensor) -> torch.Tensor:
    """
    action_pred: [B, 7]
    action_targets: [B, 7] float (same scale as dataset actions)
    """
    return F.mse_loss(action_pred.float(), action_targets.float())


def reconstruction_mse_loss(
    recon_logits: torch.Tensor,
    patch_pixels: torch.Tensor,
    patch_mask: torch.Tensor,
) -> torch.Tensor:
    """
    recon_logits, patch_pixels: [B, P, patch_dim]
    patch_mask: [B, P] bool, True where reconstruction is supervised.
    """
    pred_m = recon_logits[patch_mask]
    tgt_m = patch_pixels[patch_mask]
    if pred_m.numel() == 0:
        return recon_logits.sum() * 0.0
    return F.mse_loss(pred_m, tgt_m)


def compute_batch_losses(
    outputs: Dict[str, Any],
    action_targets: torch.Tensor,
    *,
    lambda_recon: float,
    reconstruction_enabled: bool,
) -> Dict[str, torch.Tensor]:
    """
    Returns dict with keys ``action``, ``recon`` (0 if N/A), and ``total`` for backward.
    """
    device = action_targets.device
    pred = outputs["action_logits"]
    b = action_targets.shape[0]
    exp = (b, ActionHead.NUM_ACTION_DIMS)
    if tuple(pred.shape) != exp:
        raise ValueError(
            f"action_logits must be {exp} (B, 7), got {tuple(pred.shape)}; "
            "check model ActionHead and batch size."
        )
    if tuple(action_targets.shape) != (b, ActionHead.NUM_ACTION_DIMS):
        raise ValueError(
            f"action_targets must be [B, {ActionHead.NUM_ACTION_DIMS}], got {tuple(action_targets.shape)}"
        )
    dtype = pred.dtype

    action_loss = action_mse_loss(pred, action_targets)

    recon_loss = torch.zeros((), device=device, dtype=dtype)
    if (
        reconstruction_enabled
        and outputs.get("recon_logits") is not None
        and outputs.get("patch_mask") is not None
        and outputs.get("patch_pixels") is not None
    ):
        recon_loss = reconstruction_mse_loss(
            outputs["recon_logits"],
            outputs["patch_pixels"],
            outputs["patch_mask"],
        )

    total = action_loss
    if reconstruction_enabled and lambda_recon > 0:
        total = total + lambda_recon * recon_loss

    return {"action": action_loss, "recon": recon_loss, "total": total}


def detach_loss_dict(losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Scalar floats for logging / W&B (no grad, CPU)."""
    return {k: float(v.detach().cpu()) for k, v in losses.items()}
