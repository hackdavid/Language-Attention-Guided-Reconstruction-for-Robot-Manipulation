"""Config deep-merge and repo YAML smoke."""

from pathlib import Path

import pytest

pytest.importorskip("transformers")

from code_base.config_loader import load_merged_config, load_one_config_file
from code_base.model import LAReconVLAConfigSource

_REPO = Path(__file__).resolve().parents[1]


def test_load_merged_config_order(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("training:\n  epochs: 1\n  seed: 1\n", encoding="utf-8")
    b.write_text("training:\n  epochs: 99\n", encoding="utf-8")
    cfg = load_merged_config([a, b])
    assert cfg["training"]["epochs"] == 99
    assert cfg["training"]["seed"] == 1


def test_repo_c1_full_parse():
    c1 = _REPO / "configs" / "C1.yaml"
    if not c1.is_file():
        pytest.skip("configs/C1.yaml missing")
    cfg = load_one_config_file(c1)
    assert cfg["experiment"]["condition"] == "C1"
    assert cfg["training"]["use_experiment_preset"] is False
    mc = LAReconVLAConfigSource(cfg, use_experiment_preset=cfg["training"]["use_experiment_preset"]).model_config()
    assert mc.reconstruction.enabled is False
    assert mc.masking.mode == "none"


def test_repo_c4_full_has_selected_heads():
    c4 = _REPO / "configs" / "C4.yaml"
    if not c4.is_file():
        pytest.skip("configs/C4.yaml missing")
    cfg = load_one_config_file(c4)
    heads = cfg["model"]["masking"]["selected_heads"]
    assert isinstance(heads, list) and len(heads) >= 1
    mc = LAReconVLAConfigSource(cfg, use_experiment_preset=cfg["training"]["use_experiment_preset"]).model_config()
    assert mc.masking.mode == "attention_selected"
    assert mc.masking.selected_heads is not None


def test_load_one_json(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"training": {"epochs": 3}}', encoding="utf-8")
    assert load_one_config_file(p)["training"]["epochs"] == 3
