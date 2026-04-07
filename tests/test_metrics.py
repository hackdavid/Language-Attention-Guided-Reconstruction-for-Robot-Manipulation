import torch

from code_base.metrics import action_mae_per_dim


def test_action_mae_per_dim_zero():
    b = 4
    pred = torch.randn(b, 7)
    tgt = pred.clone()
    mae = action_mae_per_dim(pred, tgt)
    assert mae.shape == (7,)
    assert torch.allclose(mae, torch.zeros(7))


def test_action_mae_per_dim_shifted():
    b = 2
    pred = torch.zeros(b, 7)
    tgt = torch.ones(b, 7)
    mae = action_mae_per_dim(pred, tgt)
    assert torch.allclose(mae, torch.ones(7))
