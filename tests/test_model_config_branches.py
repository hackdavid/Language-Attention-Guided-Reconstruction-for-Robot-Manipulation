"""Unit tests for model.py (no PaliGemma checkpoint download).

On some Windows setups, set CUDA_VISIBLE_DEVICES= before pytest if import/collection hangs.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")

from code_base.model import (
    EXPERIMENT_MODEL_PRESETS,
    ActionHead,
    LAReconVLAConfigSource,
    MAEDecoder,
    MaskingConfig,
    ModelConfig,
    ReconstructionConfig,
    apply_paligemma_trainable_rules,
    deep_merge_dict,
    ema_update_vision,
    infer_mae_spatial_from_num_image_tokens,
    model_config_from_dict,
    paligemma_language_model,
    paligemma_multi_modal_projector,
    paligemma_vision_tower,
    patchify_images,
    random_patch_mask,
    resolve_attention_layer_indices,
    saliency_topk_mask,
    _aggregate_text_to_image_saliency,
    _needs_lm_attention_for_masking,
)


def test_resolve_attention_layer_indices_list():
    assert resolve_attention_layer_indices([0, 2], 5) == [0, 2]


def test_resolve_attention_layer_indices_last_k():
    assert resolve_attention_layer_indices("last_3", 10) == [7, 8, 9]


def test_resolve_attention_layer_indices_invalid():
    with pytest.raises(ValueError):
        resolve_attention_layer_indices("last_99", 5)


def test_aggregate_saliency_shape():
    p = 4
    s = 8
    h = 2
    lt = s - p
    att = torch.randn(1, h, s, s)
    attns = (att, att)
    sal = _aggregate_text_to_image_saliency(attns, [0, 1], p, None)
    assert sal.shape == (1, p)


def test_aggregate_selected_heads():
    p = 4
    s = 10
    h = 4
    att = torch.randn(1, h, s, s)
    sal = _aggregate_text_to_image_saliency((att,), [0], p, [1, 3])
    assert sal.shape == (1, p)


def test_random_mask_fixed_k():
    m = random_patch_mask(3, 16, 5, device=torch.device("cpu"))
    assert m.shape == (3, 16)
    assert (m.sum(dim=1) == 5).all()


def test_saliency_topk():
    s = torch.tensor([[0.0, 2.0, 1.0, 0.5]])
    m = saliency_topk_mask(s, 2)
    assert m[0, 1] and m[0, 2] and not m[0, 0]


def test_patchify():
    x = torch.randn(2, 3, 224, 224)
    p = patchify_images(x, 16)
    assert p.shape == (2, 196, 16 * 16 * 3)


def test_deep_merge():
    a = {"model": {"masking": {"mode": "random"}, "a": 1}}
    b = {"model": {"masking": {"mask_ratio": 0.5}, "b": 2}}
    m = deep_merge_dict(a, b)
    assert m["model"]["masking"]["mode"] == "random"
    assert m["model"]["masking"]["mask_ratio"] == 0.5
    assert m["model"]["b"] == 2


def test_experiment_presets_cover_c1_c5():
    for k in ("C1", "C2", "C3", "C4", "C5"):
        assert k in EXPERIMENT_MODEL_PRESETS


def test_model_config_validation_recon_without_masking():
    with pytest.raises(ValueError):
        model_config_from_dict(
            {
                "experiment": {"condition": "X"},
                "model": {
                    "reconstruction": {"enabled": True},
                    "masking": {"mode": "none"},
                },
            }
        )


def test_model_config_attention_selected_requires_heads():
    with pytest.raises(ValueError):
        model_config_from_dict(
            {
                "experiment": {"condition": "X"},
                "model": {
                    "reconstruction": {"enabled": True},
                    "masking": {"mode": "attention_selected", "attention_layers": "last_2"},
                },
            }
        )


def test_model_config_ema_teacher_requires_ema():
    with pytest.raises(ValueError):
        model_config_from_dict(
            {
                "experiment": {"condition": "X"},
                "model": {
                    "reconstruction": {"enabled": True},
                    "masking": {
                        "mode": "attention_naive",
                        "mask_source": "ema_teacher",
                    },
                },
            }
        )


def test_head_selection_file():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "heads.json"
        path.write_text(json.dumps({"selected_heads": [2, 5, 7]}), encoding="utf-8")
        cfg = model_config_from_dict(
            {
                "experiment": {"condition": "C4"},
                "model": {
                    "reconstruction": {"enabled": True},
                    "masking": {
                        "mode": "attention_selected",
                        "head_selection_file": str(path),
                        "attention_layers": [0],
                    },
                },
            }
        )
        assert cfg.masking.selected_heads == [2, 5, 7]


def test_preset_merge_c4_with_heads():
    cfg = LAReconVLAConfigSource(
        {
            "experiment": {"condition": "C4"},
            "model": {"masking": {"selected_heads": [0, 1, 2]}},
        },
        use_experiment_preset=True,
    ).model_config()
    assert cfg.masking.mode == "attention_selected"
    assert cfg.reconstruction.enabled
    assert cfg.masking.selected_heads == [0, 1, 2]


def test_config_source_parse_classmethod():
    cfg = LAReconVLAConfigSource.parse(
        {
            "experiment": {"condition": "C4"},
            "model": {"masking": {"selected_heads": [1]}},
        },
        use_experiment_preset=True,
    )
    assert cfg.experiment_condition == "C4"
    assert cfg.masking.selected_heads == [1]


class _MockVisionTower(nn.Module):
    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.vision_model = nn.Module()
        self.vision_model.encoder = nn.Module()
        self.vision_model.encoder.layers = nn.ModuleList(
            [nn.Linear(8, 8) for _ in range(n_layers)]
        )
        self.vision_model.post_layernorm = nn.LayerNorm(8)


class _MockPaliGemma(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = _MockVisionTower()
        self.multi_modal_projector = nn.Linear(8, 8)
        self.language_model = nn.Linear(8, 8)


class _MockInnerPaliGemma(nn.Module):
    """HF nested layout: submodules live on ``model`` (newer transformers)."""

    def __init__(self):
        super().__init__()
        self.vision_tower = _MockVisionTower()
        self.multi_modal_projector = nn.Linear(8, 8)
        self.language_model = nn.Linear(8, 8)


class _MockPaliGemmaNested(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _MockInnerPaliGemma()


def test_paligemma_submodules_nested_hf_layout():
    m = _MockPaliGemmaNested()
    assert paligemma_vision_tower(m) is m.model.vision_tower
    assert paligemma_multi_modal_projector(m) is m.model.multi_modal_projector
    assert paligemma_language_model(m) is m.model.language_model


def test_vision_only_trainable_last_n_layers():
    m = _MockPaliGemma()
    apply_paligemma_trainable_rules(m, freeze_backbone=False, finetune_last_n_layers=2)
    trainable = {name for name, p in m.named_parameters() if p.requires_grad}
    assert any("vision_tower.vision_model.encoder.layers.2" in n for n in trainable)
    assert any("vision_tower.vision_model.encoder.layers.3" in n for n in trainable)
    assert not any("layers.0" in n for n in trainable)
    assert not m.multi_modal_projector.weight.requires_grad
    assert not m.language_model.weight.requires_grad


def test_freeze_backbone_all():
    m = _MockPaliGemma()
    apply_paligemma_trainable_rules(m, freeze_backbone=True, finetune_last_n_layers=0)
    assert not any(p.requires_grad for p in m.parameters())


def test_vision_only_trainable_nested_layout():
    m = _MockPaliGemmaNested()
    apply_paligemma_trainable_rules(m, freeze_backbone=False, finetune_last_n_layers=2)
    trainable = {name for name, p in m.named_parameters() if p.requires_grad}
    assert any("model.vision_tower.vision_model.encoder.layers.2" in n for n in trainable)
    assert any("model.vision_tower.vision_model.encoder.layers.3" in n for n in trainable)
    assert not m.model.multi_modal_projector.weight.requires_grad
    assert not m.model.language_model.weight.requires_grad


def test_ema_update():
    vt_s = _MockVisionTower()
    vt_t = _MockVisionTower()
    with torch.no_grad():
        vt_s.vision_model.encoder.layers[0].weight.fill_(1.0)
        vt_t.vision_model.encoder.layers[0].weight.fill_(0.0)
    ema_update_vision(vt_s, vt_t, decay=0.5)
    w = vt_t.vision_model.encoder.layers[0].weight
    assert torch.allclose(w, torch.full_like(w, 0.5))


def test_needs_lm_attention_for_masking():
    c_attn = ModelConfig(
        reconstruction=ReconstructionConfig(enabled=True),
        masking=MaskingConfig(mode="attention_naive"),
    )
    assert _needs_lm_attention_for_masking(c_attn)
    c_sel = ModelConfig(
        reconstruction=ReconstructionConfig(enabled=True),
        masking=MaskingConfig(mode="attention_selected", selected_heads=[0]),
    )
    assert _needs_lm_attention_for_masking(c_sel)
    c_rand = ModelConfig(
        reconstruction=ReconstructionConfig(enabled=True),
        masking=MaskingConfig(mode="random"),
    )
    assert not _needs_lm_attention_for_masking(c_rand)
    c_off = ModelConfig(
        reconstruction=ReconstructionConfig(enabled=False),
        masking=MaskingConfig(mode="attention_naive"),
    )
    assert not _needs_lm_attention_for_masking(c_off)


def test_infer_mae_spatial_from_num_image_tokens():
    assert infer_mae_spatial_from_num_image_tokens(256) == (256, 14)
    assert infer_mae_spatial_from_num_image_tokens(196) == (196, 16)


def test_infer_mae_spatial_rejects_non_square():
    with pytest.raises(ValueError, match="perfect square"):
        infer_mae_spatial_from_num_image_tokens(200)


def test_mae_decoder_shapes():
    dec = MAEDecoder(32, 16, 2, 2, num_patches=8, patch_size=16)
    feats = torch.randn(2, 8, 32)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :2] = True
    out = dec(feats, mask)
    assert out.shape == (2, 8, 16 * 16 * 3)


def test_action_head_shape():
    h = ActionHead(64)
    logits = h(torch.randn(3, 64))
    assert logits.shape == (3, 7)
