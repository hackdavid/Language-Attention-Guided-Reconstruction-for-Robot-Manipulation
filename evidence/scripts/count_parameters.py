# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""Count trainable parameters per LA-ReconVLA condition (C1-C5).

Strategy
--------
Loading PaliGemma2-3B just to count parameters is wasteful. Instead this
script:

1. Constructs the trainable, project-specific modules (MAEDecoder,
   ActionHead) using the same definitions as ``code_base.model`` and
   counts their parameters exactly.
2. Uses published PaliGemma2-3B-mix-224 component sizes (model card +
   Gemma-2 2B + SigLIP-So-400M specs) for the frozen backbone components.
3. Writes a Markdown table to ``evidence/tables/parameter_counts.md``.

Usage:
    uv run --with torch evidence/scripts/count_parameters.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]

# Architecture constants (from configs/C2.yaml ... configs/C5.yaml)
EMBED_DIM = 2304
NUM_PATCHES = 256
PATCH_SIZE = 14
DECODER_DIM = 256
DECODER_LAYERS = 4
DECODER_HEADS = 8
ACTION_HIDDEN = 512
NUM_DOFS = 7

# PaliGemma2-3B-mix-224 published component sizes (parameters).
PALIGEMMA_TOTAL = 3_032_094_976
SIGLIP_VISION_TOWER = 416_900_096
GEMMA2_LM = 2_614_341_888
MMP_LINEAR = 2_360_064
SIGLIP_BLOCK_AVG = SIGLIP_VISION_TOWER // 27


def build_mae_decoder() -> nn.Module:
    patch_dim = PATCH_SIZE * PATCH_SIZE * 3
    layer = nn.TransformerDecoderLayer(
        d_model=DECODER_DIM,
        nhead=DECODER_HEADS,
        dim_feedforward=DECODER_DIM * 4,
        batch_first=True,
        activation="gelu",
        dropout=0.0,
    )
    return nn.Sequential(
        nn.Linear(EMBED_DIM, DECODER_DIM),
        nn.TransformerDecoder(layer, num_layers=DECODER_LAYERS),
        nn.Linear(DECODER_DIM, patch_dim),
    )


def build_action_head() -> nn.Module:
    return nn.Sequential(
        nn.Linear(EMBED_DIM, ACTION_HIDDEN),
        nn.GELU(),
        nn.Linear(ACTION_HIDDEN, NUM_DOFS),
    )


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def fmt(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f} B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f} M"
    if n >= 1_000:
        return f"{n / 1e3:.2f} K"
    return str(n)


def render_markdown() -> str:
    mae = build_mae_decoder()
    mae_total = n_params(mae) + DECODER_DIM  # + [MASK] token
    action = n_params(build_action_head())
    last2_vision = 2 * SIGLIP_BLOCK_AVG

    rows = [
        ("C1 action-only",          last2_vision + action,             PALIGEMMA_TOTAL + action - (last2_vision + action),                  PALIGEMMA_TOTAL + action,                       "—"),
        ("C2 random-mask MAE",       last2_vision + action + mae_total, PALIGEMMA_TOTAL + action + mae_total - (last2_vision + action + mae_total), PALIGEMMA_TOTAL + action + mae_total, "—"),
        ("C3 naive attention MAE",   last2_vision + action + mae_total, PALIGEMMA_TOTAL + action + mae_total - (last2_vision + action + mae_total), PALIGEMMA_TOTAL + action + mae_total, "—"),
        ("C4 selected-head MAE",     last2_vision + action + mae_total, PALIGEMMA_TOTAL + action + mae_total - (last2_vision + action + mae_total), PALIGEMMA_TOTAL + action + mae_total, "—"),
        ("C5 EMA-teacher MAE",       last2_vision + action + mae_total,
         (PALIGEMMA_TOTAL + action + mae_total + SIGLIP_VISION_TOWER) - (last2_vision + action + mae_total),
         PALIGEMMA_TOTAL + action + mae_total + SIGLIP_VISION_TOWER,
         f"+{fmt(SIGLIP_VISION_TOWER)} EMA copy"),
    ]

    lines: list[str] = []
    lines.append("# Parameter counts by condition\n")
    lines.append("| Condition | Trainable | Frozen | Total | EMA / extra |")
    lines.append("|-----------|-----------|--------|-------|-------------|")
    for name, tr, fr, tot, extra in rows:
        pct = 100.0 * tr / tot
        lines.append(f"| {name} | {fmt(tr)} ({pct:.2f}%) | {fmt(fr)} | {fmt(tot)} | {extra} |")
    lines.append("")
    lines.append("## Project-specific module breakdown\n")
    lines.append("| Component | Parameters | Notes |")
    lines.append("|-----------|-----------|-------|")
    lines.append(f"| MAE decoder (4 layers, dim={DECODER_DIM}, heads={DECODER_HEADS}) | "
                 f"{fmt(mae_total)} | + 1 learnable [MASK] token ({DECODER_DIM} params) |")
    lines.append(f"| ActionHead (2-layer MLP, {EMBED_DIM}→{ACTION_HIDDEN}→{NUM_DOFS}) | "
                 f"{fmt(action)} | continuous 7-DoF Δ-action |")
    lines.append(f"| Last-2 SigLIP encoder blocks | "
                 f"{fmt(last2_vision)} | only trainable backbone parameters |")
    lines.append("")
    lines.append("## Backbone reference values (PaliGemma2-3B-mix-224)\n")
    lines.append("| Component | Parameters | Frozen? |")
    lines.append("|-----------|-----------|---------|")
    lines.append(f"| SigLIP-So-400M vision tower | {fmt(SIGLIP_VISION_TOWER)} | mostly frozen |")
    lines.append(f"| Multi-modal projector (1152→2304) | {fmt(MMP_LINEAR)} | frozen |")
    lines.append(f"| Gemma-2 2B language model | {fmt(GEMMA2_LM)} | fully frozen |")
    lines.append(f"| **PaliGemma2-3B-mix-224 total** | {fmt(PALIGEMMA_TOTAL)} | — |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    md = render_markdown()
    out_path = REPO_ROOT / "evidence" / "tables" / "parameter_counts.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    # Stdout: write a console-safe version (replace unicode arrows for cp1252).
    print(md.encode("ascii", "replace").decode("ascii"))
    print(f"Wrote {out_path.relative_to(REPO_ROOT).as_posix()}", file=sys.stderr)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
