"""
End-to-end shape smoke: LIBERO collate -> trainer contract -> losses -> one training epoch (mock backbone).

Expected dimensions: images ``[B, 3, 224, 224]``, texts length ``B``, actions ``[B, 7]`` float32.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch")
pytest.importorskip("lerobot")
pytest.importorskip("transformers")
pytest.importorskip("wandb")

from code_base.dataset_libero import (
    IMAGE_SIZE,
    KEY_ACTION,
    KEY_EPISODE,
    KEY_IMAGE,
    KEY_TASK,
    KEY_TASK_INDEX,
    KEY_WRIST,
    LiberoLoaderConfig,
    build_libero_dataloader,
    libero_collate_fn,
    libero_train_iterator_factory,
)
from code_base.losses import compute_batch_losses
from code_base.model import ActionHead
from code_base.train import (
    LAReconVLATrainer,
    default_train_config_dict,
    validate_training_batch,
)


def _rgb(seed: int) -> np.ndarray:
    a = np.zeros((16, 16, 3), dtype=np.uint8)
    a[0, 0, 0] = seed % 255
    return a


def _libero_row(i: int) -> dict:
    return {
        KEY_IMAGE: _rgb(i),
        KEY_WRIST: _rgb(i + 9),
        KEY_ACTION: [float(i + j) * 0.01 for j in range(7)],
        KEY_EPISODE: i // 2,
        KEY_TASK_INDEX: i % 3,
        KEY_TASK: f"task {i % 3}",
    }


class _FakeLAReconVLA(nn.Module):
    """Same contract as tests.test_trainer._FakeLAReconVLA (action head output [B, 7])."""

    def __init__(self, cfg):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, requires_grad=True))

    def forward(self, images, text_instructions):
        b = images.shape[0]
        logits = self.w.expand(b, ActionHead.NUM_ACTION_DIMS).float()
        return {
            "action_logits": logits,
            "vision_tokens": None,
            "recon_logits": None,
            "patch_mask": None,
            "patch_pixels": None,
            "saliency": None,
        }

    def update_ema_teacher(self) -> None:
        pass


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_libero_collate_expected_dimensions(batch_size: int):
    batch = [_libero_row(i) for i in range(batch_size)]
    images, texts, actions = libero_collate_fn(batch)
    assert images.shape == (batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert images.shape[2] == 224 and images.shape[3] == 224
    assert len(texts) == batch_size
    assert actions.shape == (batch_size, ActionHead.NUM_ACTION_DIMS)
    assert actions.dtype == torch.float32


@pytest.mark.parametrize("batch_size", [1, 3])
def test_libero_collate_validate_training_batch(batch_size: int):
    batch = [_libero_row(i) for i in range(batch_size)]
    images, texts, actions = libero_collate_fn(batch)
    validate_training_batch(images, texts, actions)


@pytest.mark.parametrize("batch_size", [2, 3])
def test_libero_dataloader_first_batch_dimensions_and_validate(batch_size: int):
    """DataLoader + ``libero_collate_fn`` (same path as real training)."""
    n = max(10, batch_size * 2)
    rows = [_libero_row(i) for i in range(n)]
    with patch("code_base.dataset_libero.LeRobotDataset", return_value=rows):
        loader = build_libero_dataloader(
            LiberoLoaderConfig(
                batch_size=batch_size,
                num_workers=0,
                shuffle=False,
                seed=0,
                skip_snapshot_download=True,
            )
        )
    images, texts, actions = next(iter(loader))
    validate_training_batch(images, texts, actions)
    assert images.shape == (batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert actions.shape == (batch_size, ActionHead.NUM_ACTION_DIMS)


def test_libero_collate_then_compute_batch_losses_backward():
    b = 2
    batch = [_libero_row(i) for i in range(b)]
    images, texts, actions = libero_collate_fn(batch)
    validate_training_batch(images, texts, actions)
    pred = torch.randn(b, ActionHead.NUM_ACTION_DIMS, requires_grad=True)
    out = {
        "action_logits": pred,
        "recon_logits": None,
        "patch_mask": None,
        "patch_pixels": None,
    }
    losses = compute_batch_losses(
        out, actions, lambda_recon=0.0, reconstruction_enabled=False
    )
    losses["total"].backward()
    assert losses["action"].shape == ()
    assert losses["total"].shape == ()


@patch("code_base.train.LAReconVLA", _FakeLAReconVLA)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_trainer_fit_libero_iterator_factory_mocked_lerobot(batch_size: int):
    n_rows = max(8, batch_size * 3)
    rows = [_libero_row(i) for i in range(n_rows)]
    with patch("code_base.dataset_libero.LeRobotDataset", return_value=rows):
        train_loader = libero_train_iterator_factory(
            LiberoLoaderConfig(
                batch_size=batch_size,
                num_workers=0,
                shuffle=False,
                seed=0,
                skip_snapshot_download=True,
            )
        )

    cfg = default_train_config_dict()
    cfg["training"]["epochs"] = 1
    cfg["training"]["device"] = "cpu"
    cfg["training"]["batch_size"] = batch_size
    cfg["training"]["val_batches"] = 0

    trainer = LAReconVLATrainer(cfg)
    hist = trainer.fit(train_loader=train_loader)
    assert len(hist["train_loss"]) == 1
    assert isinstance(hist["train_loss"][0], float)
    assert trainer._global_step >= 1
