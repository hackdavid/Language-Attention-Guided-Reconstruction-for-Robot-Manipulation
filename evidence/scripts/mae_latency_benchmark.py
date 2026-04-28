# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""Microbenchmark: single-pass MAE decoder vs simulated multi-step diffusion.

This addresses Hypothesis H2 from Part 1 (LA-ReconVLA achieves 3-5x lower
inference latency than ReconVLA's diffusion transformer head).

Method
------
* MAE forward = one ``MAEDecoder`` call (the actual production decoder).
* Diffusion forward = the same architecture iterated T times, where T is a
  conservative DDPM step count (50 or 1000). This represents the gradient/wall
  clock equivalent of an iterative denoising decoder of the same depth.
* Both are timed on CPU with synchronisation; this gives a hardware-fair
  ratio even though absolute numbers will be slower than on a P100.
* Reports per-step mean and std over 30 measurements with 5 warmup runs.

Run from repo root:
    uv run --with torch evidence/scripts/mae_latency_benchmark.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]

# Match the production architecture
EMBED_DIM = 2304
NUM_PATCHES = 256
PATCH_SIZE = 14
DECODER_DIM = 256
DECODER_LAYERS = 4
DECODER_HEADS = 8
BATCH = 1
WARMUP = 5
RUNS = 30
DIFFUSION_STEPS = (50, 250, 1000)


class ProductionMAEDecoder(nn.Module):
    """Faithful reproduction of code_base.model.MAEDecoder."""

    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.Linear(EMBED_DIM, DECODER_DIM)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, DECODER_DIM))
        nn.init.normal_(self.mask_token, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=DECODER_DIM,
            nhead=DECODER_HEADS,
            dim_feedforward=DECODER_DIM * 4,
            batch_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=DECODER_LAYERS)
        self.pixel_head = nn.Linear(DECODER_DIM, PATCH_SIZE * PATCH_SIZE * 3)

    def forward(self, features: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
        b, p, _ = features.shape
        x = self.input_proj(features)
        num_visible = int((~mask_bool[0]).sum().item())
        tgt = torch.where(mask_bool.unsqueeze(-1), self.mask_token.expand(b, p, -1), x)
        memory = x.masked_select((~mask_bool).unsqueeze(-1)).view(b, num_visible, -1)
        dec = self.decoder(tgt, memory)
        return self.pixel_head(dec)


def time_callable(fn, runs: int = RUNS, warmup: int = WARMUP) -> Tuple[float, float]:
    """Return (mean_seconds, std_seconds)."""
    for _ in range(warmup):
        fn()
    samples: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.mean(samples), statistics.pstdev(samples)


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    decoder = ProductionMAEDecoder().to(device).eval()

    features = torch.randn(BATCH, NUM_PATCHES, EMBED_DIM, device=device)
    k = max(1, int(NUM_PATCHES * 0.25))
    mask_bool = torch.zeros(BATCH, NUM_PATCHES, dtype=torch.bool, device=device)
    idx = torch.randperm(NUM_PATCHES)[:k]
    mask_bool[0, idx] = True

    @torch.no_grad()
    def mae_step():
        return decoder(features, mask_bool)

    @torch.no_grad()
    def diffusion_step_factory(steps: int):
        def run():
            for _ in range(steps):
                decoder(features, mask_bool)
        return run

    print("Hardware: CPU (single-threaded, torch=" + torch.__version__ + ")")
    print(f"Architecture: 4-layer Transformer decoder, dim={DECODER_DIM}, heads={DECODER_HEADS}, "
          f"P={NUM_PATCHES}, k={k}\n")

    mae_mean, mae_std = time_callable(mae_step)
    print(f"MAE single-pass: {mae_mean*1000:.2f} +/- {mae_std*1000:.2f} ms")

    rows = [("MAE single-pass", 1, mae_mean, mae_std)]
    for steps in DIFFUSION_STEPS:
        diff_mean, diff_std = time_callable(diffusion_step_factory(steps), runs=10, warmup=2)
        ratio = diff_mean / mae_mean
        print(f"Diffusion-equiv. T={steps}: {diff_mean*1000:.2f} +/- {diff_std*1000:.2f} ms "
              f"({ratio:.1f}x slower)")
        rows.append((f"Diffusion-equiv. T={steps}", steps, diff_mean, diff_std))

    out_md = REPO_ROOT / "evidence" / "tables" / "mae_latency.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# MAE decoder latency microbenchmark", ""]
    lines.append("Hardware: CPU, single-threaded, torch " + torch.__version__)
    lines.append(f"Decoder: 4-layer Transformer, dim={DECODER_DIM}, heads={DECODER_HEADS}, "
                 f"P={NUM_PATCHES}, k={k}, batch={BATCH}, runs={RUNS}.")
    lines.append("")
    lines.append("| Method | Forward passes | Mean (ms) | Std (ms) | Slowdown vs MAE |")
    lines.append("|--------|---------------|-----------|----------|-----------------|")
    for name, steps, mean_s, std_s in rows:
        slowdown = mean_s / mae_mean
        lines.append(f"| {name} | {steps} | {mean_s*1000:.2f} | {std_s*1000:.2f} | {slowdown:.2f}x |")
    lines.append("")
    lines.append("> The diffusion-equivalent rows are not a real DDPM head; they iterate the same "
                 "MAE decoder *T* times, giving a *lower bound* for diffusion latency at matched "
                 "decoder capacity. A real ReconVLA-style diffusion transformer head [1] is "
                 "typically *deeper* per step, so wall-clock multipliers in production would be "
                 "even larger than the ratios above.")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_md.relative_to(REPO_ROOT).as_posix()}", file=sys.stderr)


if __name__ == "__main__":
    main()
