# LA-ReconVLA

Vision–language–action training with a PaliGemma backbone, LIBERO/LeRobot data, and optional reconstruction / masking experiments. Configs live under `configs/`; training code under `code_base/`.

## Requirements

- Python 3.10+
- PyTorch with CUDA (recommended for `training.device: cuda` in the YAML)
- Hugging Face account (dataset download; optional token for gated assets)
- [Weights & Biases](https://wandb.ai) account if you enable logging in config

Install PyTorch from [pytorch.org](https://pytorch.org) for your platform, then:

```bash
pip install -r requirements-training.txt
```

## Quick start (local)

From the **repository root** (so `code_base` is importable):

```bash
# Dataset: set cache root (optional). Default is ./data/libero_spatial_image
set LIBERO_DATASET_ROOT=%CD%\data\libero_spatial_image
set HF_TOKEN=your_hf_token_if_needed

# Weights & Biases (if logging.wandb.enabled is true in your YAML)
set WANDB_API_KEY=your_wandb_api_key

python train.py --config configs/C1.yaml
```

Equivalent:

```bash
python -m code_base.train --config configs/C1.yaml
```

Disable W&B regardless of YAML:

```bash
python train.py --config configs/C1.yaml --no-wandb
```

Merge overrides (right-hand file wins):

```bash
python train.py --config configs/C1.yaml configs/my_local_overrides.local.yaml
```

Use a **gitignored** `configs/*.local.yaml` for API keys or machine paths (see `.gitignore`).

## Configs

See [`configs/README.md`](configs/README.md). Files are safe to commit: **no API keys**. Set `WANDB_API_KEY` (and optionally `HF_TOKEN`) in the environment.

## Google Colab (GPU)

1. **Runtime → Change runtime type → GPU** (T4 ~15 GB). Check with `!nvidia-smi`.
2. Configs use **`training.device: cuda`** and **`training.mixed_precision: true`** (AMP on GPU).
3. **`data.libero`** is read automatically: training uses the real LIBERO dataloader (not dummy data). Tune **`data.libero.batch_size`** (try `4`, drop to `2` if you see CUDA OOM).

Run from the repo root after cloning. Set secrets **before** launching training so `wandb` and `huggingface_hub` pick them up.

```python
# --- Clone ---
# !git clone https://github.com/YOUR_USERNAME/la-reconvla.git
# %cd la-reconvla

# --- Install PyTorch for your runtime (example: CUDA 12.x; check https://pytorch.org) ---
# !pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
# !pip install --quiet -r requirements-training.txt

# --- Secrets (pick one approach) ---
import os

# A) Paste for the session (do not save notebook with real keys)
os.environ["WANDB_API_KEY"] = "paste_wandb_key_here"
os.environ["HF_TOKEN"] = "paste_hf_token_here"  # for LeRobot / Hub dataset download

# B) Colab “Secrets” (recommended): store WANDB_API_KEY and HF_TOKEN in the sidebar, then:
# from google.colab import userdata
# os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
# os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

# --- Dataset location (optional; default under ./data/...) ---
# os.environ["LIBERO_DATASET_ROOT"] = "/content/dataset"

# --- Train (same flags as local) ---
# !python train.py --config configs/C1.yaml
# !python train.py --config configs/C2.yaml --no-wandb
```

W&B details:

- If `logging.wandb.enabled: true`, the training code calls `wandb.login` using `WANDB_API_KEY` (or YAML `api_key` / `key` if you must; prefer env and never commit keys).
- Omit `run_name` in YAML to get an automatic unique run name per launch.
- Offline logging: `os.environ["WANDB_MODE"] = "offline"` then sync later with `wandb sync`.

## Tests

```bash
pip install pytest
pytest tests/ -q
```

## Layout

| Path | Role |
|------|------|
| `train.py` | CLI shim → `code_base.train` |
| `code_base/train.py` | Training loop, argparse |
| `code_base/dataset_libero.py` | LIBERO snapshot + `LeRobotDataset` + collate |
| `code_base/model.py` | LAReconVLA |
| `configs/*.yaml` | Experiment definitions |
