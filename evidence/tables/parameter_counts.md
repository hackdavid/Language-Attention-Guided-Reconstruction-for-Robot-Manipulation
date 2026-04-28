# Parameter counts by condition

| Condition | Trainable | Frozen | Total | EMA / extra |
|-----------|-----------|--------|-------|-------------|
| C1 action-only | 32.07 M (1.06%) | 3.00 B | 3.03 B | — |
| C2 random-mask MAE | 37.02 M (1.22%) | 3.00 B | 3.04 B | — |
| C3 naive attention MAE | 37.02 M (1.22%) | 3.00 B | 3.04 B | — |
| C4 selected-head MAE | 37.02 M (1.22%) | 3.00 B | 3.04 B | — |
| C5 EMA-teacher MAE | 37.02 M (1.07%) | 3.42 B | 3.46 B | +416.90 M EMA copy |

## Project-specific module breakdown

| Component | Parameters | Notes |
|-----------|-----------|-------|
| MAE decoder (4 layers, dim=256, heads=8) | 4.96 M | + 1 learnable [MASK] token (256 params) |
| ActionHead (2-layer MLP, 2304→512→7) | 1.18 M | continuous 7-DoF Δ-action |
| Last-2 SigLIP encoder blocks | 30.88 M | only trainable backbone parameters |

## Backbone reference values (PaliGemma2-3B-mix-224)

| Component | Parameters | Frozen? |
|-----------|-----------|---------|
| SigLIP-So-400M vision tower | 416.90 M | mostly frozen |
| Multi-modal projector (1152→2304) | 2.36 M | frozen |
| Gemma-2 2B language model | 2.61 B | fully frozen |
| **PaliGemma2-3B-mix-224 total** | 3.03 B | — |
