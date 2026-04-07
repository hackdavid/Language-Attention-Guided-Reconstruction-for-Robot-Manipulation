import pytest
import torch

pytest.importorskip("transformers")

from code_base.losses import compute_batch_losses


def test_compute_batch_losses_action_only():
    b = 2
    pred = torch.randn(b, 7, requires_grad=True)
    tgt = torch.randn(b, 7)
    out = {
        "action_logits": pred,
        "recon_logits": None,
        "patch_mask": None,
        "patch_pixels": None,
    }
    losses = compute_batch_losses(
        out, tgt, lambda_recon=0.5, reconstruction_enabled=False
    )
    losses["total"].backward()
    assert losses["action"].ndim == 0
    assert torch.allclose(losses["recon"], torch.zeros_like(losses["recon"]))


def test_compute_batch_losses_with_recon():
    b, p, d = 2, 4, 12
    pred = torch.randn(b, 7, requires_grad=True)
    recon = torch.randn(b, p, d, requires_grad=True)
    mask = torch.zeros(b, p, dtype=torch.bool)
    mask[:, :2] = True
    tgt = torch.randn(b, 7)
    pix = torch.randn(b, p, d)
    out = {
        "action_logits": pred,
        "recon_logits": recon,
        "patch_mask": mask,
        "patch_pixels": pix,
    }
    losses = compute_batch_losses(out, tgt, lambda_recon=0.5, reconstruction_enabled=True)
    losses["total"].backward()


def test_compute_batch_losses_rejects_logits_shape():
    b = 2
    tgt = torch.randn(b, 7)
    out = {
        "action_logits": torch.randn(b, 7, 4),
        "recon_logits": None,
        "patch_mask": None,
        "patch_pixels": None,
    }
    with pytest.raises(ValueError, match="action_logits must be"):
        compute_batch_losses(out, tgt, lambda_recon=0.0, reconstruction_enabled=False)
