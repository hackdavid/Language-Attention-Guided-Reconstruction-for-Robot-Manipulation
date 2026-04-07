"""
LIBERO Spatial image dataset (hub: ``lerobot/libero_spatial_image``).

Data is fetched with :func:`huggingface_hub.snapshot_download` into a fixed local root
(override with env ``LIBERO_DATASET_ROOT``), then loaded with :class:`lerobot.datasets.lerobot_dataset.LeRobotDataset`
on a **hard-coded episode subset** (see :data:`LIBERO_EPISODES`).

Schema: ``observation.images.image``, ``observation.images.wrist_image``, ``action``,
``episode_index``, ``task_index``, ``task``. Training batches are ``(images, texts, actions)``
with images ``[B, 3, 224, 224]`` (main + wrist fused side-by-side), ``actions`` ``[B, 7]``,
and each text equal to the row's ``task`` string (no extra episode / task_index suffix).

Auth: set ``HF_TOKEN`` or run ``huggingface-cli login`` (do not embed tokens in code).

**CLI check**: ``python -m code_base.dataset_libero`` or ``python code_base/dataset_libero.py`` (see :func:`main`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import snapshot_download

def _logging_utils_standalone() -> Any:
    """
    Load sibling ``logging_utils.py`` without importing the ``code_base`` package.

    When this file is run as ``python code_base/dataset_libero.py``, ``__package__`` is
    ``None`` and ``from code_base.logging_utils`` would execute ``code_base/__init__.py``
    (checkpoint, train, torch, etc.). Script mode only needs logging.
    """
    import importlib.util

    name = "code_base.logging_utils"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parent / "logging_utils.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load logging utils from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if __package__ is None:
    get_logger = _logging_utils_standalone().get_logger
else:
    from .logging_utils import get_logger

import argparse
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_log = get_logger(__name__)

# --- Hub identity and fixed episode subset (training split) ---
LIBERO_REPO_ID = "lerobot/libero_spatial_image"

LIBERO_EPISODES: Tuple[int, ...] = (
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    10,
    13,
    15,
    16,
    23,
    27,
    30,
    31,
    37,
    40,
    41,
    43,
    48,
    52,
    53,
    68,
    72,
    73,
    75,
    76,
    77,
    78,
    79,
    80,
    82,
    88,
    98,
    99,
    104,
    110,
    112,
    113,
    127,
    133,
    135,
    143,
    154,
    155,
    157,
    159,
    160,
    168,
    169,
    172,
    173,
    180,
    182,
    185,
    187,
    192,
    195,
    200,
    201,
    204,
    208,
    209,
    211,
    212,
    213,
    220,
    221,
    224,
    233,
    235,
    237,
    241,
    244,
    258,
    260,
    263,
    264,
    265,
    273,
    276,
    278,
    282,
    283,
    290,
    300,
    301,
    303,
    314,
    315,
    316,
    320,
    321,
    325,
    330,
    333,
    334,
    337,
    343,
    344,
    345,
    346,
    350,
    351,
    352,
    358,
    359,
    363,
    364,
    365,
    374,
    379,
    384,
    385,
    386,
    390,
    394,
    400,
    402,
    405,
    416,
    419,
    423,
    424,
    426,
    428,
    431,
)

IMAGE_SIZE = 224
HALF_W = IMAGE_SIZE // 2
FULL_H = IMAGE_SIZE

KEY_IMAGE = "observation.images.image"
KEY_WRIST = "observation.images.wrist_image"
KEY_ACTION = "action"
KEY_EPISODE = "episode_index"
KEY_TASK_INDEX = "task_index"
KEY_TASK = "task"


def libero_local_dir() -> Path:
    """Root directory for ``snapshot_download`` + :class:`LeRobotDataset` (created if missing)."""
    env = os.environ.get("LIBERO_DATASET_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "data" / "libero_spatial_image").resolve()


def download_libero_snapshot(
    local_dir: Optional[Path] = None,
    *,
    token: Optional[str] = None,
) -> Path:
    """
    Download the full hub dataset into ``local_dir`` (idempotent; uses the Hub cache protocol).

    Token: pass explicitly, or set ``HF_TOKEN``, or use a prior ``huggingface-cli login``.
    """
    root = local_dir if local_dir is not None else libero_local_dir()
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    tok = token if token is not None else os.environ.get("HF_TOKEN")
    _log.info("LIBERO: snapshot_download repo_id=%r -> %s", LIBERO_REPO_ID, root)
    snapshot_download(
        repo_id=LIBERO_REPO_ID,
        repo_type="dataset",
        local_dir=str(root),
        token=tok,
    )
    return root


@dataclass
class LiberoLoaderConfig:
    """Dataloader options; repo, episodes, and fusion are fixed in this module."""

    batch_size: int = 4
    num_workers: int = 0
    shuffle: bool = True
    seed: Optional[int] = 42
    local_root: Optional[str] = None
    skip_snapshot_download: bool = False
    # True with CUDA + DataLoader pin_memory for non_blocking H2D copies in the trainer.
    pin_memory: bool = False


def _scalar_py(x: Any) -> Any:
    """Turn a 0-d tensor, ndarray, or scalar into a Python scalar or str."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.item() if x.size == 1 else x.tolist()
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]
    return x


def _value_to_pil_rgb(value: Any) -> Image.Image:
    """Decode a frame field to a PIL RGB image (supports tensors, ndarray, PIL, HF image dict)."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, torch.Tensor):
        t = value.detach().cpu()
        if t.dim() == 3 and t.shape[0] in (1, 3):
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)
            if t.dtype in (torch.float32, torch.float16, torch.bfloat16):
                mx = float(t.max()) if t.numel() else 0.0
                if mx <= 1.0 + 1e-3:
                    t = (t.clamp(0, 1) * 255).to(torch.uint8)
                else:
                    t = t.clamp(0, 255).to(torch.uint8)
            else:
                t = t.to(torch.uint8)
            arr = t.permute(1, 2, 0).contiguous().numpy()
            return Image.fromarray(arr).convert("RGB")
        if t.dim() == 3 and t.shape[-1] == 3:
            arr = t.numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
        raise TypeError(f"Bad image tensor shape: {tuple(t.shape)}")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
        raise TypeError(f"Unsupported image dict keys: {value.keys()}")
    if isinstance(value, np.ndarray):
        arr = value if value.dtype == np.uint8 else np.clip(value, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            raise ValueError("Expected HxWx3 image")
        return Image.fromarray(arr).convert("RGB")
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(value)}")


def fuse_main_wrist_chw_float(main: Any, wrist: Any) -> torch.Tensor:
    """
    Resize main and wrist each to ``112×224``, concatenate horizontally to ``224×224`` RGB,
    return ``[3, 224, 224]`` float in ``[0, 1]`` (PaliGemma-friendly).
    """
    p_main = _value_to_pil_rgb(main).resize((HALF_W, FULL_H), Image.BILINEAR)
    p_wrist = _value_to_pil_rgb(wrist).resize((HALF_W, FULL_H), Image.BILINEAR)
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
    canvas.paste(p_main, (0, 0))
    canvas.paste(p_wrist, (HALF_W, 0))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def format_instruction(row: Dict[str, Any]) -> str:
    """Return the dataset ``task`` string as the language instruction (verbatim, stripped)."""
    return str(_scalar_py(row[KEY_TASK])).strip()


def action_to_float7(row: Dict[str, Any]) -> torch.Tensor:
    """Read ``action`` as a length-7 float vector."""
    act = row[KEY_ACTION]
    if isinstance(act, torch.Tensor):
        a = act.detach().cpu().float().reshape(-1)
    elif isinstance(act, (list, tuple)):
        a = torch.tensor(act, dtype=torch.float32).reshape(-1)
    else:
        a = torch.as_tensor(np.asarray(act), dtype=torch.float32).reshape(-1)
    if a.numel() != 7:
        raise ValueError(f"Expected 7-d action, got shape {tuple(a.shape)}")
    return a


def libero_collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
    """
    Collate a list of frame dicts into trainer batch format.

    Applies fusion, text formatting, and action stacking. Each element of ``batch`` must
    expose the hub schema keys listed in the module docstring.
    """
    images = torch.stack(
        [fuse_main_wrist_chw_float(row[KEY_IMAGE], row[KEY_WRIST]) for row in batch],
        dim=0,
    )
    texts = [format_instruction(row) for row in batch]
    actions = torch.stack([action_to_float7(row) for row in batch], dim=0)
    return images, texts, actions


class LiberoMapRows(Dataset):
    """
    Map dataset: ``__getitem__(i)`` returns the i-th frame dict (no preprocessing).

    Preprocessing is deferred to :func:`libero_collate_fn`.
    """

    def __init__(self, source: Any) -> None:
        self._src = source

    def __len__(self) -> int:
        return len(self._src)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self._src[index]
        return dict(row) if not isinstance(row, dict) else row


def build_libero_dataloader(cfg: Optional[LiberoLoaderConfig] = None) -> DataLoader:
    """
    Build a :class:`torch.utils.data.DataLoader` over local LIBERO data.

    Runs :func:`download_libero_snapshot` unless ``cfg.skip_snapshot_download`` is True
    (tests or when the snapshot is already on disk). Then wraps
    ``LeRobotDataset(repo_id, root=..., episodes=LIBERO_EPISODES, download_videos=True)``.

    Batch tuples are ``(images, texts, actions)`` per :func:`libero_collate_fn`.
    """
    c = cfg or LiberoLoaderConfig()
    root = Path(c.local_root).expanduser().resolve() if c.local_root else libero_local_dir()
    if not c.skip_snapshot_download:
        download_libero_snapshot(root)
    _log.info(
        "LIBERO dataloader: LeRobotDataset repo_id=%r root=%s episodes=%s",
        LIBERO_REPO_ID,
        root,
        len(LIBERO_EPISODES),
    )
    lerobot_ds = LeRobotDataset(
        repo_id=LIBERO_REPO_ID,
        root=root,
        episodes=list(LIBERO_EPISODES),
        download_videos=True,
    )
    ds = LiberoMapRows(lerobot_ds)
    g: Optional[torch.Generator] = None
    if c.shuffle and c.seed is not None:
        g = torch.Generator()
        g.manual_seed(int(c.seed))
    return DataLoader(
        ds,
        batch_size=c.batch_size,
        shuffle=c.shuffle,
        num_workers=c.num_workers,
        collate_fn=libero_collate_fn,
        generator=g,
        pin_memory=c.pin_memory,
    )


def libero_train_iterator_factory(
    cfg: Optional[LiberoLoaderConfig] = None,
) -> Any:
    """
    Return ``lambda: iter(loader)`` for :meth:`LAReconVLATrainer.fit`, sharing one
    :class:`DataLoader` so each epoch gets a new iterator (fresh shuffle when enabled).
    """
    loader = build_libero_dataloader(cfg)
    return lambda: iter(loader)


def is_hf_hub_offline() -> bool:
    """True if ``HF_HUB_OFFLINE`` env suggests no Hub access."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def _assert_libero_batch(
    images: torch.Tensor,
    texts: List[str],
    actions: torch.Tensor,
) -> None:
    """Same tensor contract as :func:`code_base.train.validate_training_batch` (no trainer import)."""
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(f"images must be [B, 3, H, W], got {tuple(images.shape)}")
    b = images.shape[0]
    if len(texts) != b:
        raise ValueError(f"len(texts)={len(texts)} != batch size B={b}")
    if tuple(actions.shape) != (b, 7):
        raise ValueError(f"actions must be [B, 7], got {tuple(actions.shape)}")
    if images.shape[2] != IMAGE_SIZE or images.shape[3] != IMAGE_SIZE:
        raise ValueError(f"expected H=W={IMAGE_SIZE}, got {tuple(images.shape)}")


def _run_like_training_loop(
    cfg: LiberoLoaderConfig,
    *,
    max_batches_per_epoch: int,
    epochs: int,
    label: str,
) -> int:
    """
    Mirror :meth:`LAReconVLATrainer.fit`: each epoch call ``train_loader()`` from
    :func:`libero_train_iterator_factory`, then ``for batch in train_loader()`` unpacking
    ``(images, texts, actions)``. Stops each epoch after ``max_batches_per_epoch`` batches
    (or sooner if the dataset is shorter).
    """
    if max_batches_per_epoch < 1:
        raise ValueError("max_batches_per_epoch must be >= 1")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    factory = libero_train_iterator_factory(cfg)
    total = 0
    for ep in range(epochs):
        n_ep = 0
        for batch in factory():
            print(batch)
            images, texts, actions = batch
            _assert_libero_batch(images, texts, actions)
            n_ep += 1
            total += 1
            if ep == 0 and n_ep == 1:
                t0 = texts[0] if texts else ""
                preview = (t0[:120] + "…") if len(t0) > 120 else t0
                _log.info(
                    "%s: first batch ep=%s B=%s images=%s actions=%s sample_text=%r",
                    label,
                    ep + 1,
                    images.shape[0],
                    tuple(images.shape),
                    tuple(actions.shape),
                    preview,
                )
            if n_ep >= max_batches_per_epoch:
                break
        _log.info("%s: epoch %s/%s batches_this_epoch=%s total_so_far=%s", label, ep + 1, epochs, n_ep, total)
    _log.info("%s: finished %s epochs, %s batches total", label, epochs, total)
    return total


def main(argv: Optional[List[str]] = None) -> int:
    """
    Exercise the LIBERO dataloader (snapshot + LeRobot subset) like training.

    Run from repo root::

        python -m code_base.dataset_libero
        python -m code_base.dataset_libero --max-batches 20 --epochs 2
        python -m code_base.dataset_libero --skip-snapshot  # data already under LIBERO_DATASET_ROOT
    """
    if __package__ is None:
        _lu = _logging_utils_standalone()
        configure_training_logging = _lu.configure_training_logging
        log_level_from_env = _lu.log_level_from_env
    else:
        from .logging_utils import configure_training_logging, log_level_from_env

    parser = argparse.ArgumentParser(description="LIBERO dataloader check (snapshot + LeRobot, training-style loop).")
    parser.add_argument("--batch-size", type=int, default=4, help="Per batch (default 4).")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=10,
        help="Per epoch: stop after this many batches.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of outer epochs (new iterator per epoch, same as trainer).",
    )
    parser.add_argument(
        "--skip-snapshot",
        action="store_true",
        help="Do not call snapshot_download (expect dataset already at LIBERO_DATASET_ROOT or default data dir).",
    )
    parser.add_argument(
        "--local-root",
        default=None,
        help="Override local dataset directory (else LIBERO_DATASET_ROOT or ./data/libero_spatial_image).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Override TRAINING_LOG_LEVEL (default: INFO).",
    )
    args = parser.parse_args(argv)

    import logging as _logging

    if args.log_level:
        level = {"DEBUG": _logging.DEBUG, "INFO": _logging.INFO, "WARNING": _logging.WARNING, "ERROR": _logging.ERROR}[
            args.log_level
        ]
    else:
        level = log_level_from_env()
    configure_training_logging(level=level)

    if is_hf_hub_offline() and not args.skip_snapshot:
        _log.error("HF_HUB_OFFLINE is set; use --skip-snapshot if data is already local.")
        return 2

    cfg = LiberoLoaderConfig(
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        seed=42,
        local_root=args.local_root,
        skip_snapshot_download=args.skip_snapshot,
    )
    _run_like_training_loop(
        cfg,
        max_batches_per_epoch=args.max_batches,
        epochs=args.epochs,
        label="LIBERO (snapshot + LeRobotDataset)",
    )
    _log.info("LIBERO dataloader main: finished OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
