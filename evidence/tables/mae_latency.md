# MAE decoder latency microbenchmark

Hardware: CPU, single-threaded, torch 2.11.0+cpu
Decoder: 4-layer Transformer, dim=256, heads=8, P=256, k=64, batch=1, runs=30.

| Method | Forward passes | Mean (ms) | Std (ms) | Slowdown vs MAE |
|--------|---------------|-----------|----------|-----------------|
| MAE single-pass | 1 | 51.90 | 6.67 | 1.00x |
| Diffusion-equiv. T=50 | 50 | 3022.32 | 247.53 | 58.23x |
| Diffusion-equiv. T=250 | 250 | 15199.04 | 282.45 | 292.83x |
| Diffusion-equiv. T=1000 | 1000 | 63583.85 | 6977.93 | 1225.03x |

> The diffusion-equivalent rows are not a real DDPM head; they iterate the same MAE decoder *T* times, giving a *lower bound* for diffusion latency at matched decoder capacity. A real ReconVLA-style diffusion transformer head [1] is typically *deeper* per step, so wall-clock multipliers in production would be even larger than the ratios above.