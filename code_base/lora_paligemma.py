"""
LoRA injection for PaliGemma via PEFT.

YAML supplies ``target_modules`` explicitly (e.g. ``q_proj``, ``v_proj``); PEFT matches
any submodule whose **last name component** equals one of these strings anywhere in the
model tree (language model, vision tower, multi-modal projector, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Sequence

from .logging_utils import get_logger

_log = get_logger(__name__)


@dataclass
class LoRACoreConfig:
    """Subset of PEFT ``LoraConfig`` fields; ``target_modules`` is user-controlled."""

    enabled: bool = False
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    target_modules: List[str] = field(default_factory=list)


def paligemma_root_module(pg: Any) -> Any:
    """
    Return the inner Hugging Face module (unwrap ``PeftModel`` / tuners).

    Used so helpers like ``paligemma_vision_tower`` see the same object graph PEFT patched.
    """
    cur: Any = pg
    for _ in range(8):
        gbm = getattr(cur, "get_base_model", None)
        if not callable(gbm):
            break
        nxt = gbm()
        if nxt is cur:
            break
        cur = nxt
    return cur


def apply_lora_to_paligemma(pg: Any, cfg: LoRACoreConfig) -> Any:
    """
    Wrap ``PaliGemmaForConditionalGeneration`` with LoRA using explicit ``target_modules``.

    Raises:
        ImportError: if ``peft`` is not installed.
        ValueError: if ``target_modules`` is empty.
    """
    if not cfg.enabled:
        raise ValueError("apply_lora_to_paligemma called with cfg.enabled=False")
    if not cfg.target_modules:
        raise ValueError("lora.target_modules must be a non-empty list when lora.enabled is true")

    try:
        from peft import LoraConfig, get_peft_model  # type: ignore[import-untyped]
        from peft.utils import TaskType  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "peft is required for LoRA. Install with: pip install peft"
        ) from e

    bias = str(cfg.bias).lower()
    if bias not in ("none", "all", "lora_only"):
        raise ValueError(f"lora.bias must be one of none, all, lora_only; got {cfg.bias!r}")

    modules: List[str] = [str(m).strip() for m in cfg.target_modules if str(m).strip()]
    if not modules:
        raise ValueError("lora.target_modules must contain at least one non-empty module name")

    lora_config = LoraConfig(
        r=int(cfg.r),
        lora_alpha=int(cfg.lora_alpha),
        lora_dropout=float(cfg.lora_dropout),
        bias=bias,
        target_modules=modules,
        task_type=TaskType.CAUSAL_LM,
    )
    out = get_peft_model(pg, lora_config)
    n_trainable = sum(p.numel() for p in out.parameters() if p.requires_grad)
    _log.info(
        "LoRA applied: target_modules=%s r=%s alpha=%s trainable_params=%s",
        modules,
        cfg.r,
        cfg.lora_alpha,
        n_trainable,
    )
    return out


def parse_lora_config_from_dict(d: Any) -> LoRACoreConfig:
    """Build :class:`LoRACoreConfig` from YAML ``model.lora`` dict (or empty)."""
    if d is None:
        return LoRACoreConfig()
    if not isinstance(d, dict):
        raise TypeError("model.lora must be a mapping or null")

    raw_targets = d.get("target_modules", [])
    if isinstance(raw_targets, str):
        modules = [raw_targets.strip()] if raw_targets.strip() else []
    elif isinstance(raw_targets, Sequence):
        modules = [str(x).strip() for x in raw_targets if str(x).strip()]
    else:
        raise TypeError("lora.target_modules must be a string or list of strings")

    return LoRACoreConfig(
        enabled=bool(d.get("enabled", False)),
        r=int(d.get("r", 8)),
        lora_alpha=int(d.get("lora_alpha", d.get("alpha", 16))),
        lora_dropout=float(d.get("lora_dropout", d.get("dropout", 0.0))),
        bias=str(d.get("bias", "none")),
        target_modules=modules,
    )
