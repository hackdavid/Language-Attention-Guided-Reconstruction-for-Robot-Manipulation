"""Trainer smoke tests (mock LAReconVLA — no HuggingFace download)."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("transformers")
pytest.importorskip("wandb")
import torch
import torch.nn as nn

from code_base.train import (
    LAReconVLATrainer,
    TrainingSettings,
    default_train_config_dict,
    validate_training_batch,
)


class _FakeLAReconVLA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, requires_grad=True))

    def forward(self, images, text_instructions):
        b = images.shape[0]
        logits = self.w.expand(b, 7).float()
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


def test_parse_training_defaults():
    t = LAReconVLATrainer._parse_training({})
    assert isinstance(t, TrainingSettings)
    assert t.epochs >= 1 and t.batch_size >= 1


def test_validate_training_batch_ok():
    b = 2
    img = torch.rand(b, 3, 224, 224)
    texts = ["a", "b"]
    act = torch.randn(b, 7, dtype=torch.float32)
    validate_training_batch(img, texts, act)


def test_validate_training_batch_rejects_mismatch():
    with pytest.raises(ValueError, match="len\\(texts\\)"):
        validate_training_batch(
            torch.rand(2, 3, 224, 224),
            ["only_one"],
            torch.randn(2, 7),
        )


def test_validate_training_batch_rejects_non_float_actions():
    with pytest.raises(ValueError, match="float dtype"):
        validate_training_batch(
            torch.rand(1, 3, 224, 224),
            ["x"],
            torch.randint(0, 4, (1, 7), dtype=torch.long),
        )


@patch("code_base.train.LAReconVLA", _FakeLAReconVLA)
def test_trainer_fit_dummy_cpu():
    cfg = default_train_config_dict()
    cfg["training"]["epochs"] = 1
    cfg["training"]["batches_per_epoch"] = 2
    cfg["training"]["device"] = "cpu"
    cfg["training"]["batch_size"] = 2
    trainer = LAReconVLATrainer(cfg)
    hist = trainer.fit()
    assert len(hist["train_loss"]) == 1
    assert isinstance(hist["train_loss"][0], float)


@patch("code_base.train.LAReconVLA", _FakeLAReconVLA)
def test_trainer_custom_loader_factory():
    cfg = default_train_config_dict()
    cfg["training"]["epochs"] = 1
    cfg["training"]["device"] = "cpu"
    cfg["training"]["batch_size"] = 1

    def loader():
        def gen():
            for _ in range(3):
                yield (
                    torch.rand(1, 3, 224, 224),
                    ["answer en test ."],
                    torch.randn(1, 7),
                )

        return gen()

    trainer = LAReconVLATrainer(cfg)
    hist = trainer.fit(train_loader=loader)
    assert len(hist["train_loss"]) == 1


@patch("code_base.train.LAReconVLA", _FakeLAReconVLA)
def test_eval_step_with_action_mae():
    cfg = default_train_config_dict()
    cfg["training"]["device"] = "cpu"
    cfg["training"]["batch_size"] = 2
    t = LAReconVLATrainer(cfg)
    images = torch.rand(2, 3, 224, 224)
    texts = ["a", "b"]
    actions = torch.randn(2, 7)
    losses, sum_abs, bsz = t.eval_step_with_action_mae(images, texts, actions)
    assert bsz == 2
    assert "total" in losses
    assert sum_abs.shape == (7,)


@patch("code_base.train.LAReconVLA", _FakeLAReconVLA)
def test_trainer_wandb_mocked():
    cfg = default_train_config_dict()
    cfg["training"]["epochs"] = 1
    cfg["training"]["batches_per_epoch"] = 2
    cfg["training"]["device"] = "cpu"
    cfg["training"]["batch_size"] = 2
    cfg["training"]["val_batches"] = 1
    cfg["logging"]["wandb"]["enabled"] = True
    mock_run = MagicMock(id="wandb-test")
    with patch("wandb.init", return_value=mock_run), patch("wandb.log"), patch("wandb.finish"):
        trainer = LAReconVLATrainer(cfg)
        trainer.fit()


@patch("code_base.train.LAReconVLA", _FakeLAReconVLA)
def test_trainer_checkpoint_resume(tmp_path):
    cfg = default_train_config_dict()
    cfg["training"]["epochs"] = 2
    cfg["training"]["batches_per_epoch"] = 1
    cfg["training"]["device"] = "cpu"
    cfg["training"]["checkpoint_dir"] = str(tmp_path / "ckpt")

    t1 = LAReconVLATrainer(cfg)
    t1.fit()
    assert (tmp_path / "ckpt" / "latest.pt").is_file()
    assert (tmp_path / "ckpt" / "best.pt").is_file()

    cfg2 = default_train_config_dict()
    cfg2["training"].update(cfg["training"])
    cfg2["training"]["epochs"] = 4
    cfg2["training"]["resume_from"] = "latest"
    t2 = LAReconVLATrainer(cfg2)
    assert t2._resume_next_epoch == 2
    t2.fit()
    assert t2._global_step > 0
