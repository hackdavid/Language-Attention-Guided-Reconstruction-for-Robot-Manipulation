"""W&B wrapper tests (mocked SDK; no network)."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("wandb")

from code_base.wandb_training import (
    WandbExperimentLogger,
    WandbSettings,
    experiment_run_name,
    experiment_tags,
    parse_wandb_settings,
    unique_wandb_run_name,
)


def test_parse_wandb_defaults():
    s = parse_wandb_settings({})
    assert s.enabled is False
    assert s.project == "la-reconvla"


def test_parse_wandb_api_key_aliases():
    s = parse_wandb_settings({"logging": {"wandb": {"enabled": True, "key": " k-from-yaml "}}})
    assert s.api_key == "k-from-yaml"
    s2 = parse_wandb_settings({"logging": {"wandb": {"api_key": "x"}}})
    assert s2.api_key == "x"


def test_unique_wandb_run_name_format():
    n = unique_wandb_run_name({"experiment": {"name": "C1_action_only"}})
    assert n.startswith("C1_action_only_")
    parts = n.split("_")
    assert len(parts[-1]) == 8  # hex suffix


def test_parse_wandb_from_config():
    cfg = {
        "logging": {
            "wandb": {
                "enabled": True,
                "project": "myproj",
                "tags": ["C4", "seed42"],
                "log_train_every_n_steps": 5,
            }
        }
    }
    s = parse_wandb_settings(cfg)
    assert s.enabled is True
    assert s.project == "myproj"
    assert "C4" in s.tags
    assert s.log_train_every_n_steps == 5


def test_experiment_meta_helpers():
    assert experiment_run_name({"experiment": {"name": "run_a"}}) == "run_a"
    assert experiment_tags({"experiment": {"condition": "C1", "name": "n"}}) == ["C1", "n"]


def test_wandb_logger_calls_sdk():
    mock_run = MagicMock(id="test-run-id")
    with (
        patch("wandb.init", return_value=mock_run) as mock_init,
        patch("wandb.log") as mock_log,
        patch("wandb.finish") as mock_finish,
        patch("wandb.login"),
    ):
        w = WandbExperimentLogger(
            WandbSettings(enabled=True, project="la-reconvla"),
            {"experiment": {"condition": "C1"}, "training": {"epochs": 1}},
        )
        w.start()
        assert w.active
        mock_init.assert_called_once()
        assert mock_init.call_args.kwargs.get("name")
        w.log_train_step(2, {"total": 1.0, "action": 0.5, "recon": 0.1})
        mock_log.assert_called_once()
        w.log_epoch_end(
            2,
            epoch_1based=1,
            mean_train_loss=0.9,
            mean_val_loss=0.8,
            val_action_mae_per_dim=[0.1] * 7,
            best_metric=0.8,
            checkpoint_saved_best=True,
        )
        assert mock_log.call_count == 2
        w.finish()
        mock_finish.assert_called_once()


def test_wandb_login_called_with_config_api_key():
    mock_run = MagicMock(id="id")
    with patch("wandb.login") as mock_login, patch("wandb.init", return_value=mock_run):
        WandbExperimentLogger(
            WandbSettings(enabled=True, project="la-reconvla", api_key="cfg-secret"),
            {"experiment": {"name": "n"}},
        ).start()
    mock_login.assert_called_once_with(key="cfg-secret", relogin=True)


def test_wandb_explicit_run_name_passthrough():
    mock_run = MagicMock(id="id")
    with patch("wandb.login"), patch("wandb.init", return_value=mock_run) as mock_init:
        WandbExperimentLogger(
            WandbSettings(enabled=True, project="p", run_name="my_manual_run"),
            {"experiment": {"name": "ignored_for_name"}},
        ).start()
    assert mock_init.call_args.kwargs["name"] == "my_manual_run"
