"""Tests for PEFT LoRA helpers (optional ``peft`` for apply smoke test)."""

from __future__ import annotations

import pytest

from code_base.lora_paligemma import (
    LoRACoreConfig,
    apply_lora_to_paligemma,
    paligemma_root_module,
    parse_lora_config_from_dict,
)
from code_base.model import model_config_from_dict


def test_parse_lora_empty():
    c = parse_lora_config_from_dict(None)
    assert not c.enabled
    assert c.target_modules == []


def test_parse_lora_target_modules_list():
    c = parse_lora_config_from_dict(
        {
            "enabled": True,
            "r": 8,
            "alpha": 24,
            "target_modules": ["q_proj", "v_proj"],
        }
    )
    assert c.enabled
    assert c.r == 8
    assert c.lora_alpha == 24
    assert c.target_modules == ["q_proj", "v_proj"]


def test_parse_lora_single_string_target():
    c = parse_lora_config_from_dict({"enabled": False, "target_modules": "q_proj"})
    assert c.target_modules == ["q_proj"]


def test_model_config_lora_enabled_requires_targets():
    with pytest.raises(ValueError, match="target_modules"):
        model_config_from_dict(
            {
                "experiment": {"condition": "X"},
                "model": {"lora": {"enabled": True, "target_modules": []}},
            }
        )


def test_paligemma_root_module_no_wrap():
    import torch.nn as nn

    m = nn.Linear(3, 3)
    assert paligemma_root_module(m) is m


def test_apply_lora_tiny_llama_smoke():
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    LlamaConfig = transformers.LlamaConfig
    LlamaForCausalLM = transformers.LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
    )
    base = LlamaForCausalLM(cfg)
    wrapped = apply_lora_to_paligemma(
        base,
        LoRACoreConfig(
            enabled=True,
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    assert any(p.requires_grad for p in wrapped.parameters())
    assert not all(p.requires_grad for p in wrapped.parameters())
