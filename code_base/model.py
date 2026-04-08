"""
LA-ReconVLA-style model: PaliGemma backbone (vision-only fine-tuning), optional MAE decoder,
continuous action head. Behavior is driven by C1–C5 experiment config (masking / reconstruction / EMA).

When reconstruction is enabled, ``num_patches`` / ``patch_size`` are aligned to the loaded
checkpoint's ``num_image_tokens`` and 224×224 inputs so MAE masks match ``image_hidden_states``.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from .logging_utils import get_logger

MaskingMode = Literal["none", "random", "attention_naive", "attention_selected"]
MaskSource = Literal["student", "ema_teacher"]
AttentionHeadsSpec = Union[Literal["all"], List[int]]


@dataclass
class ReconstructionConfig:
    enabled: bool = False
    lambda_recon: float = 0.0
    decoder_layers: int = 4
    decoder_dim: int = 256
    decoder_heads: int = 8


@dataclass
class MaskingConfig:
    mode: MaskingMode = "none"
    mask_ratio: float = 0.25
    attention_heads: AttentionHeadsSpec = "all"
    attention_layers: Union[str, List[int]] = "last_3"
    selected_heads: Optional[List[int]] = None
    head_selection_file: Optional[str] = None
    mask_source: MaskSource = "student"
    topology: str = "default"


@dataclass
class EMAConfig:
    enabled: bool = False
    decay: float = 0.999


@dataclass
class BackboneConfig:
    model_id: str = "google/paligemma-3b-pt-224"
    torch_dtype: str = "bfloat16"
    device_map: Optional[Union[str, Dict[str, Any]]] = None


@dataclass
class ModelConfig:
    experiment_condition: str = "C1"
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    freeze_backbone: bool = True
    finetune_last_n_layers: int = 0
    num_patches: int = 196
    patch_size: int = 16
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)

    def __post_init__(self) -> None:
        if self.masking.head_selection_file and self.masking.selected_heads is None:
            with open(self.masking.head_selection_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            heads = data.get("selected_heads") or data.get("heads")
            if heads is None:
                raise ValueError("head_selection_file must contain 'selected_heads' or 'heads' list")
            self.masking.selected_heads = list(map(int, heads))


def _needs_lm_attention_for_masking(cfg: "ModelConfig") -> bool:
    """True when forward uses ``output_attentions`` (SDPA does not support this — use eager)."""
    if not cfg.reconstruction.enabled:
        return False
    return cfg.masking.mode in ("attention_naive", "attention_selected")


def _ensure_paligemma_eager_attention(model: nn.Module) -> None:
    """
    PyTorch SDPA attention returns no per-head weights; saliency masking needs eager attention.
    """
    setter = getattr(model, "set_attn_implementation", None)
    if callable(setter):
        try:
            setter("eager")
            return
        except Exception as e:
            log = get_logger(__name__)
            log.warning("set_attn_implementation('eager') failed (%s); patching config", e)
    c = getattr(model, "config", None)
    if c is None:
        return
    for sub in (c, getattr(c, "text_config", None)):
        if sub is not None and hasattr(sub, "attn_implementation"):
            setattr(sub, "attn_implementation", "eager")


def _dtype_from_string(name: str) -> torch.dtype:
    m = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return m[name.lower()]


def resolve_attention_layer_indices(
    attention_layers: Union[str, List[int]], num_attn_layers: int
) -> List[int]:
    """Map `attention_layers` config to 0-based indices into `outputs.attentions`."""
    if isinstance(attention_layers, list):
        idxs = [int(i) for i in attention_layers]
        for i in idxs:
            if i < 0 or i >= num_attn_layers:
                raise ValueError(f"attention layer index {i} out of range [0, {num_attn_layers})")
        return idxs
    s = str(attention_layers).strip()
    m = re.match(r"last_(\d+)$", s, re.I)
    if m:
        k = int(m.group(1))
        if k <= 0 or k > num_attn_layers:
            raise ValueError(f"last_{k} invalid for num_attn_layers={num_attn_layers}")
        return list(range(num_attn_layers - k, num_attn_layers))
    raise ValueError(
        f"attention_layers must be a list of ints or a string like 'last_3', got {attention_layers!r}"
    )


def _aggregate_text_to_image_saliency(
    attentions: Tuple[torch.Tensor, ...],
    layer_indices: List[int],
    num_image_tokens: int,
    head_indices: Optional[List[int]],
) -> torch.Tensor:
    """
    PaliGemma merges image+text in the LM; use text query rows × image key columns.
    attentions[ℓ]: [B, H, S, S] with first num_image_tokens positions = patches.
    Returns saliency [B, num_image_tokens] (higher = more attended from text).
    """
    acc: Optional[torch.Tensor] = None
    for li in layer_indices:
        attn = attentions[li]
        if attn.shape[-1] < num_image_tokens or attn.shape[-2] < num_image_tokens + 1:
            raise ValueError(
                f"Unexpected attention shape {tuple(attn.shape)} for num_image_tokens={num_image_tokens}"
            )
        block = attn[:, :, num_image_tokens:, :num_image_tokens]
        if head_indices is not None:
            block = block[:, head_indices, :, :]
        sal = block.mean(dim=1).mean(dim=-2)
        acc = sal if acc is None else acc + sal
    assert acc is not None
    return acc / len(layer_indices)


def random_patch_mask(batch_size: int, num_patches: int, k: int, device: torch.device) -> torch.Tensor:
    """Boolean mask [B, P], True = masked (reconstruct). Exactly k True per row."""
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
    for b in range(batch_size):
        perm = torch.randperm(num_patches, device=device)[:k]
        mask[b, perm] = True
    return mask


def deep_merge_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dict `a` with `b`; values in `b` override `a`."""
    out = dict(a)
    for key, val in b.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge_dict(cast(Dict[str, Any], out[key]), val)
        else:
            out[key] = val
    return out


EXPERIMENT_MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "C1": {
        "model": {
            "reconstruction": {"enabled": False, "lambda_recon": 0.0},
            "masking": {"mode": "none"},
        }
    },
    "C2": {
        "model": {
            "reconstruction": {"enabled": True, "lambda_recon": 0.5},
            "masking": {"mode": "random", "mask_ratio": 0.25},
        }
    },
    "C3": {
        "model": {
            "reconstruction": {"enabled": True},
            "masking": {"mode": "attention_naive", "attention_heads": "all", "attention_layers": "last_3"},
        }
    },
    "C4": {
        "model": {
            "reconstruction": {"enabled": True},
            "masking": {"mode": "attention_selected", "attention_layers": "last_3"},
        }
    },
    "C5": {
        "model": {
            "reconstruction": {"enabled": True},
            "masking": {
                "mode": "attention_selected",
                "attention_layers": "last_3",
                "mask_source": "ema_teacher",
            },
            "ema": {"enabled": True, "decay": 0.999},
        }
    },
}


def saliency_topk_mask(saliency: torch.Tensor, k: int) -> torch.Tensor:
    """saliency [B, P] -> boolean mask True on top-k highest per row."""
    _vals, idx = saliency.topk(k, dim=-1)
    mask = torch.zeros_like(saliency, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def patchify_images(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    images: [B, 3, H, W] with H,W divisible by patch_size
    returns: [B, num_patches, patch_size**2 * 3]
    """
    b, c, h, w = images.shape
    if h % patch_size or w % patch_size:
        raise ValueError(f"Image size ({h},{w}) not divisible by patch_size={patch_size}")
    gh, gw = h // patch_size, w // patch_size
    x = images.reshape(b, c, gh, patch_size, gw, patch_size)
    x = torch.einsum("bchpwq->bhwcpq", x)
    x = x.reshape(b, gh * gw, c * patch_size * patch_size)
    return x


# PaliGemma 224 checkpoints use a square SigLIP grid; MAE must use the same P and patch_size.
DEFAULT_MAE_IMAGE_SIZE = 224


def infer_mae_spatial_from_num_image_tokens(
    num_image_tokens: int, *, image_size: int = DEFAULT_MAE_IMAGE_SIZE
) -> Tuple[int, int]:
    """
    Map backbone ``num_image_tokens`` to ``(num_patches, patch_size)`` for square inputs.

    ``num_patches`` equals ``num_image_tokens``; ``patch_size = image_size // sqrt(num_patches)``.
    """
    if num_image_tokens <= 0:
        raise ValueError(f"num_image_tokens must be positive, got {num_image_tokens}")
    g = int(math.isqrt(num_image_tokens))
    if g * g != num_image_tokens:
        raise ValueError(
            f"num_image_tokens={num_image_tokens} is not a perfect square; "
            "cannot infer a square patch grid for MAE."
        )
    if image_size % g != 0:
        raise ValueError(
            f"image_size={image_size} not divisible by grid side g={g} "
            f"(num_image_tokens={num_image_tokens})"
        )
    return num_image_tokens, image_size // g


class ActionHead(nn.Module):
    NUM_ACTION_DIMS = 7

    def __init__(self, embed_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.NUM_ACTION_DIMS),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class MAEDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        decoder_dim: int,
        num_layers: int,
        num_heads: int,
        num_patches: int,
        patch_size: int,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.patch_dim = patch_size * patch_size * 3
        self.input_proj = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            batch_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.pixel_head = nn.Linear(decoder_dim, self.patch_dim)

    def forward(self, features: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
        """
        features: [B, P, embed_dim]; mask_bool [B, P] True = masked.
        Returns predictions for all positions [B, P, patch_dim]; use mask for loss.
        """
        b, p, _ = features.shape
        x = self.input_proj(features)
        if mask_bool.shape != (b, p):
            raise ValueError("mask_bool must match features batch and num_patches")
        num_visible = int((~mask_bool[0]).sum().item())
        if not ((~mask_bool).sum(dim=1) == num_visible).all():
            raise ValueError("MAEDecoder expects the same number of visible patches per batch item")
        tgt = torch.where(mask_bool.unsqueeze(-1), self.mask_token.expand(b, p, -1), x)
        memory = x.masked_select((~mask_bool).unsqueeze(-1)).view(b, num_visible, -1)
        dec = self.decoder(tgt, memory)
        return self.pixel_head(dec)


def _vision_encoder_layers(vision_tower: nn.Module) -> Optional[nn.ModuleList]:
    vm = getattr(vision_tower, "vision_model", None)
    if vm is None:
        return None
    enc = getattr(vm, "encoder", None)
    if enc is None:
        return None
    return getattr(enc, "layers", None)


def apply_paligemma_trainable_rules(
    model: nn.Module, freeze_backbone: bool, finetune_last_n_layers: int
) -> None:
    """
    If freeze_backbone: freeze entire PaliGemma (vision + projector + LM).
    Else: freeze language_model and multi_modal_projector; train vision_tower only,
    optionally restricted to the last N SigLIP encoder blocks (+ post_layernorm).
    """
    for p in model.parameters():
        p.requires_grad = False
    if freeze_backbone:
        return
    if not hasattr(model, "vision_tower"):
        raise AttributeError("Expected PaliGemmaForConditionalGeneration.vision_tower")
    vt = model.vision_tower
    layers = _vision_encoder_layers(vt)
    if finetune_last_n_layers <= 0 or layers is None:
        for p in vt.parameters():
            p.requires_grad = True
    else:
        n = len(layers)
        start = max(0, n - finetune_last_n_layers)
        for i in range(start, n):
            for p in layers[i].parameters():
                p.requires_grad = True
        vm = getattr(vt, "vision_model", None)
        if vm is not None and hasattr(vm, "post_layernorm"):
            for p in vm.post_layernorm.parameters():
                p.requires_grad = True


def clone_vision_tower(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module)


@torch.no_grad()
def ema_update_vision(
    student_vt: nn.Module, teacher_vt: nn.Module, decay: float
) -> None:
    s = dict(student_vt.named_parameters())
    for name, t_param in teacher_vt.named_parameters():
        if name not in s:
            continue
        t_param.data.mul_(decay).add_(s[name].data, alpha=1.0 - decay)


def _paligemma_text_with_image_token(text_instructions: Sequence[str]) -> List[str]:
    """
    PaliGemmaProcessor expects a ``<image>`` placeholder at the start of each string when
    passing ``images=`` (one image per text); avoids HF "infer special tokens" warnings.
    """
    out: List[str] = []
    for raw in text_instructions:
        t = str(raw)
        if t.lstrip().startswith("<image>"):
            out.append(t)
        else:
            out.append(f"<image>\n{t}" if t else "<image>")
    return out


class PaliGemmaBackbone(nn.Module):
    """Loads PaliGemma processor + model; exposes projected image tokens and optional LM attentions."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        dtype = _dtype_from_string(cfg.backbone.torch_dtype)
        load_kw: Dict[str, Any] = {"torch_dtype": dtype}
        if cfg.backbone.device_map is not None:
            load_kw["device_map"] = cfg.backbone.device_map
        if _needs_lm_attention_for_masking(cfg):
            load_kw["attn_implementation"] = "eager"
        self.processor = PaliGemmaProcessor.from_pretrained(cfg.backbone.model_id)
        self.paligemma = PaliGemmaForConditionalGeneration.from_pretrained(
            cfg.backbone.model_id, **load_kw
        )
        if _needs_lm_attention_for_masking(cfg):
            _ensure_paligemma_eager_attention(self.paligemma)
            log = get_logger(__name__)
            log.info("Attention masking: using attn_implementation=eager (required for output_attentions).")
        apply_paligemma_trainable_rules(
            self.paligemma, cfg.freeze_backbone, cfg.finetune_last_n_layers
        )

    @property
    def num_image_tokens(self) -> int:
        return int(self.paligemma.config.text_config.num_image_tokens)

    @property
    def lm_hidden_size(self) -> int:
        return int(self.paligemma.config.text_config.hidden_size)

    def forward_paligemma(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        paligemma_module: Optional[Any] = None,
    ) -> Any:
        pg = paligemma_module or self.paligemma
        return pg(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
        )

    def build_inputs(
        self, images: torch.Tensor, text_instructions: Sequence[str], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        texts = _paligemma_text_with_image_token(text_instructions)
        proc = self.processor(text=texts, images=list(images), return_tensors="pt", padding=True)
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in proc.items()}


def forward_language_model_attentions(
    pg: Any,
    vision_tower: nn.Module,
    *,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Run the (frozen) language model with image embeddings from a given vision tower
    and return the attention tuple (one tensor per layer).
    """
    emb_dtype = pg.get_input_embeddings().weight.dtype
    pv = pixel_values.to(device=input_ids.device, dtype=emb_dtype)
    image_outputs = vision_tower(pv)
    selected_image_feature = image_outputs.last_hidden_state
    image_features = pg.multi_modal_projector(selected_image_feature)
    image_features = image_features / (pg.config.text_config.hidden_size**0.5)
    inputs_embeds = pg.get_input_embeddings()(input_ids)
    special_image_mask = (input_ids == pg.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    device = input_ids.device
    past_key_values = None
    past_seen_tokens = 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=device
    )
    position_ids = cache_position.unsqueeze(0) + 1
    causal_mask = pg._update_causal_mask(
        attention_mask,
        token_type_ids,
        past_key_values,
        cache_position,
        inputs_embeds,
        is_training=False,
    )
    lm_out = pg.language_model(
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
        cache_position=cache_position,
    )
    if lm_out.attentions is None:
        raise RuntimeError("language_model returned attentions=None")
    return lm_out.attentions


class LAReconVLA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = PaliGemmaBackbone(cfg)
        self._align_mae_spatial_to_backbone()
        d = self.backbone.lm_hidden_size
        self.action_head = ActionHead(d)
        rc = cfg.reconstruction
        self.mae: Optional[MAEDecoder] = None
        if rc.enabled:
            self.mae = MAEDecoder(
                embed_dim=d,
                decoder_dim=rc.decoder_dim,
                num_layers=rc.decoder_layers,
                num_heads=rc.decoder_heads,
                num_patches=cfg.num_patches,
                patch_size=cfg.patch_size,
            )
        self.teacher_vision: Optional[nn.Module] = None
        if cfg.ema.enabled and cfg.masking.mask_source == "ema_teacher":
            self.teacher_vision = clone_vision_tower(self.backbone.paligemma.vision_tower)
            for p in self.teacher_vision.parameters():
                p.requires_grad = False
            self.teacher_vision.eval()

    def _align_mae_spatial_to_backbone(self) -> None:
        """When reconstruction is on, force ``num_patches`` / ``patch_size`` to match image tokens."""
        if not self.cfg.reconstruction.enabled:
            return
        n_vis = self.backbone.num_image_tokens
        n_p, p_sz = infer_mae_spatial_from_num_image_tokens(
            n_vis, image_size=DEFAULT_MAE_IMAGE_SIZE
        )
        if self.cfg.num_patches != n_p or self.cfg.patch_size != p_sz:
            log = get_logger(__name__)
            log.warning(
                "MAE spatial config aligned to PaliGemma: num_patches %s -> %s, patch_size %s -> %s "
                "(backbone num_image_tokens=%s, image_size=%s)",
                self.cfg.num_patches,
                n_p,
                self.cfg.patch_size,
                p_sz,
                n_vis,
                DEFAULT_MAE_IMAGE_SIZE,
            )
        self.cfg.num_patches = n_p
        self.cfg.patch_size = p_sz

    def forward(
        self,
        images: torch.Tensor,
        text_instructions: Sequence[str],
    ) -> Dict[str, Any]:
        """
        Forward pass only: logits and reconstruction tensors. Loss is computed in training
        (see ``code_base.losses``).

        Returns:
            ``action_logits`` [B, 7] (continuous action prediction), ``vision_tokens`` [B, P, D],
            and when reconstruction is enabled: ``recon_logits`` [B, P, patch_dim],
            ``patch_mask`` [B, P] bool, ``patch_pixels`` [B, P, patch_dim] targets,
            optional ``saliency`` [B, P]. Otherwise recon fields are ``None``.
        """
        device = images.device
        b = images.shape[0]
        inputs = self.backbone.build_inputs(images, text_instructions, device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"].to(dtype=self.backbone.paligemma.dtype)
        token_type_ids = inputs.get("token_type_ids")

        need_attn = self.cfg.reconstruction.enabled and self.cfg.masking.mode in (
            "attention_naive",
            "attention_selected",
        )
        use_teacher_mask = (
            self.cfg.ema.enabled
            and self.cfg.masking.mask_source == "ema_teacher"
            and self.teacher_vision is not None
            and need_attn
        )

        out_main = self.backbone.forward_paligemma(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            output_attentions=need_attn,
        )
        img_tokens = out_main.image_hidden_states
        if img_tokens is None:
            raise RuntimeError("PaliGemma did not return image_hidden_states (pixel_values missing?)")
        pooled = img_tokens.mean(dim=1)
        action_logits = self.action_head(pooled)

        result: Dict[str, Any] = {
            "action_logits": action_logits,
            "vision_tokens": img_tokens,
        }

        mask_bool: Optional[torch.Tensor] = None
        saliency: Optional[torch.Tensor] = None

        k = max(1, int(self.cfg.num_patches * self.cfg.masking.mask_ratio))

        if self.cfg.reconstruction.enabled and self.mae is not None:
            patch_pixels_tensor = patchify_images(images, self.cfg.patch_size)
            mask_bool = torch.zeros(b, self.cfg.num_patches, dtype=torch.bool, device=device)
            mode = self.cfg.masking.mode

            if mode == "none" or mode == "random":
                mask_bool = random_patch_mask(b, self.cfg.num_patches, k, device)
            elif mode in ("attention_naive", "attention_selected"):
                saliency_src = out_main
                if use_teacher_mask:
                    with torch.no_grad():
                        self.teacher_vision.eval()
                        attns = forward_language_model_attentions(
                            self.backbone.paligemma,
                            self.teacher_vision,
                            pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                        )
                        saliency_src = SimpleNamespace(attentions=attns)
                if not hasattr(saliency_src, "attentions") or saliency_src.attentions is None:
                    raise RuntimeError("Attention masking requested but attentions are missing")
                attns = saliency_src.attentions
                layer_ix = resolve_attention_layer_indices(
                    self.cfg.masking.attention_layers, len(attns)
                )
                head_ix: Optional[List[int]] = None
                if mode == "attention_selected":
                    heads = self.cfg.masking.selected_heads
                    if heads is None:
                        raise ValueError("attention_selected requires masking.selected_heads or head_selection_file")
                    head_ix = list(map(int, heads))
                elif self.cfg.masking.attention_heads != "all":
                    if isinstance(self.cfg.masking.attention_heads, list):
                        head_ix = list(map(int, self.cfg.masking.attention_heads))
                saliency = _aggregate_text_to_image_saliency(
                    attns,
                    layer_ix,
                    self.backbone.num_image_tokens,
                    head_ix,
                )
                mask_bool = saliency_topk_mask(saliency, k)
            else:
                raise ValueError(f"Unknown masking.mode {mode!r}")

            preds = self.mae(img_tokens, mask_bool)
            result["recon_logits"] = preds
            result["patch_mask"] = mask_bool
            result["patch_pixels"] = patch_pixels_tensor.to(dtype=preds.dtype)
            result["saliency"] = saliency
        else:
            result["recon_logits"] = None
            result["patch_mask"] = None
            result["patch_pixels"] = None
            result["saliency"] = None

        return result

    @torch.no_grad()
    def update_ema_teacher(self) -> None:
        if self.teacher_vision is None:
            return
        ema_update_vision(
            self.backbone.paligemma.vision_tower,
            self.teacher_vision,
            self.cfg.ema.decay,
        )


class LAReconVLAConfigSource:
    """
    Parses a nested experiment/config dict (e.g. merged YAML) into ``ModelConfig``
    for ``LAReconVLA``.

    Example::

        source = LAReconVLAConfigSource(merged_yaml, use_experiment_preset=True)
        cfg = source.model_config()
        model = LAReconVLA(cfg)
    """

    def __init__(
        self,
        config_dict: Dict[str, Any],
        *,
        use_experiment_preset: bool = False,
    ) -> None:
        self._config_dict = dict(config_dict)
        self._use_experiment_preset = use_experiment_preset

    @classmethod
    def parse(
        cls,
        config_dict: Dict[str, Any],
        *,
        use_experiment_preset: bool = False,
    ) -> ModelConfig:
        """Convenience: build and return ``ModelConfig`` in one call."""
        return cls(config_dict, use_experiment_preset=use_experiment_preset).model_config()

    def model_config(self) -> ModelConfig:
        """Merge presets (optional), parse nested keys, validate, return ``ModelConfig``."""
        src = dict(self._config_dict)
        if self._use_experiment_preset:
            exp = src.get("experiment") or {}
            cond = str(exp.get("condition", "C1")).upper()
            preset = EXPERIMENT_MODEL_PRESETS.get(cond)
            if preset is not None:
                src = deep_merge_dict(preset, src)
        exp = src.get("experiment") or {}
        m = src.get("model") or {}
        cond = exp.get("condition", src.get("experiment_condition", "C1"))
        bb = m.get("backbone", "google/paligemma-3b-pt-224")
        if isinstance(bb, str):
            bb_cfg = BackboneConfig(model_id=bb)
        else:
            bb_cfg = BackboneConfig(
                model_id=bb.get("model_id", "google/paligemma-3b-pt-224"),
                torch_dtype=bb.get("torch_dtype", "bfloat16"),
                device_map=bb.get("device_map"),
            )
        rec = m.get("reconstruction") or {}
        mask = m.get("masking") or {}
        ema = m.get("ema") or {}
        cfg = ModelConfig(
            experiment_condition=str(cond),
            backbone=bb_cfg,
            freeze_backbone=bool(m.get("freeze_backbone", True)),
            finetune_last_n_layers=int(m.get("finetune_last_n_layers", 0)),
            num_patches=int(m.get("num_patches", 196)),
            patch_size=int(m.get("patch_size", 16)),
            reconstruction=ReconstructionConfig(
                enabled=bool(rec.get("enabled", False)),
                lambda_recon=float(rec.get("lambda_recon", 0.0)),
                decoder_layers=int(rec.get("decoder_layers", 4)),
                decoder_dim=int(rec.get("decoder_dim", 256)),
                decoder_heads=int(rec.get("decoder_heads", 8)),
            ),
            masking=MaskingConfig(
                mode=mask.get("mode", "none"),
                mask_ratio=float(mask.get("mask_ratio", 0.25)),
                attention_heads=mask.get("attention_heads", "all"),
                attention_layers=mask.get("attention_layers", "last_3"),
                selected_heads=mask.get("selected_heads"),
                head_selection_file=mask.get("head_selection_file"),
                mask_source=mask.get("mask_source", "student"),
                topology=str(mask.get("topology", "default")),
            ),
            ema=EMAConfig(
                enabled=bool(ema.get("enabled", False)),
                decay=float(ema.get("decay", 0.999)),
            ),
        )
        cfg.__post_init__()
        self._validate(cfg)
        return cfg

    @staticmethod
    def _validate(cfg: ModelConfig) -> None:
        if cfg.reconstruction.enabled:
            if cfg.masking.mode == "none":
                raise ValueError(
                    "reconstruction.enabled with masking.mode=none is invalid; use C1 or disable reconstruction"
                )
            if cfg.masking.mode == "attention_selected" and cfg.masking.selected_heads is None:
                raise ValueError("attention_selected requires selected_heads or head_selection_file")
        if cfg.masking.mask_source == "ema_teacher" and not cfg.ema.enabled:
            raise ValueError("mask_source=ema_teacher requires ema.enabled=true")


def model_config_from_dict(
    d: Dict[str, Any], *, use_experiment_preset: bool = False
) -> ModelConfig:
    """Backward-compatible alias for :meth:`LAReconVLAConfigSource.parse`."""
    return LAReconVLAConfigSource.parse(d, use_experiment_preset=use_experiment_preset)


def build_model(
    config_dict: Dict[str, Any], *, use_experiment_preset: bool = False
) -> LAReconVLA:
    cfg = LAReconVLAConfigSource(config_dict, use_experiment_preset=use_experiment_preset).model_config()
    return LAReconVLA(cfg)
