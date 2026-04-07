"""Tests for minimal LIBERO loader (in-memory rows; no hub unless integration)."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("torch")
pytest.importorskip("lerobot")

from code_base.dataset_libero import (
    KEY_ACTION,
    KEY_EPISODE,
    KEY_IMAGE,
    KEY_TASK,
    KEY_TASK_INDEX,
    KEY_WRIST,
    LiberoLoaderConfig,
    LiberoMapRows,
    action_to_float7,
    build_libero_dataloader,
    format_instruction,
    fuse_main_wrist_chw_float,
    libero_collate_fn,
    libero_train_iterator_factory,
)


def _cfg_test(**kwargs):
    base = dict(
        batch_size=2,
        shuffle=False,
        seed=0,
        skip_snapshot_download=True,
    )
    base.update(kwargs)
    return LiberoLoaderConfig(**base)


def _rgb(seed: int) -> np.ndarray:
    a = np.zeros((16, 16, 3), dtype=np.uint8)
    a[0, 0, 0] = seed % 255
    return a


def _row(i: int) -> dict:
    return {
        KEY_IMAGE: _rgb(i),
        KEY_WRIST: _rgb(i + 9),
        KEY_ACTION: [float(i + j) * 0.01 for j in range(7)],
        KEY_EPISODE: i // 2,
        KEY_TASK_INDEX: i % 3,
        KEY_TASK: f"task {i % 3}",
    }


def test_fuse_and_instruction_and_action():
    r = _row(0)
    img = fuse_main_wrist_chw_float(r[KEY_IMAGE], r[KEY_WRIST])
    assert img.shape == (3, 224, 224)
    t = format_instruction(r)
    assert t == "task 0"
    a = action_to_float7(r)
    assert a.shape == (7,) and a.dtype == torch.float32


def test_libero_collate_fn():
    batch = [_row(0), _row(1)]
    images, texts, actions = libero_collate_fn(batch)
    assert images.shape == (2, 3, 224, 224)
    assert len(texts) == 2
    assert actions.shape == (2, 7)


def test_build_dataloader_mocked_lerobot():
    rows = [_row(i) for i in range(5)]

    with patch("code_base.dataset_libero.LeRobotDataset", return_value=rows):
        loader = build_libero_dataloader(_cfg_test(batch_size=2))
    batch = next(iter(loader))
    images, texts, actions = batch
    assert images.shape == (2, 3, 224, 224)
    assert actions.shape == (2, 7)


def test_libero_train_iterator_factory_mocked():
    rows = [_row(i) for i in range(4)]
    with patch("code_base.dataset_libero.LeRobotDataset", return_value=rows):
        factory = libero_train_iterator_factory(_cfg_test(batch_size=2))
        it = factory()
        images, texts, actions = next(it)
    assert images.shape[0] == 2


def test_build_lerobot_load_raises():
    with patch(
        "code_base.dataset_libero.LeRobotDataset",
        side_effect=ImportError("simulated hub/lerobot failure"),
    ):
        with pytest.raises(ImportError):
            build_libero_dataloader(_cfg_test())


def test_libero_map_rows():
    rows = [_row(i) for i in range(3)]
    ds = LiberoMapRows(rows)
    assert len(ds) == 3
    assert KEY_TASK in ds[0]
