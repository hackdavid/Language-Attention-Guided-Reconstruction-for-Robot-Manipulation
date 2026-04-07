import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from code_base.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    build_checkpoint_payload,
    load_checkpoint,
    load_training_state,
)


def test_build_and_load_checkpoint():
    m = nn.Linear(3, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    payload = build_checkpoint_payload(
        next_epoch=3,
        global_step=100,
        best_metric=0.42,
        model_state_dict=m.state_dict(),
        optimizer_state_dict=opt.state_dict(),
        scaler_state_dict=None,
        config_dict={"a": 1},
    )
    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.pt"
        torch.save(payload, path)
        raw = load_checkpoint(path, map_location="cpu")
        assert raw["next_epoch"] == 3
        assert raw["global_step"] == 100

        m2 = nn.Linear(3, 2)
        opt2 = torch.optim.AdamW(m2.parameters(), lr=9.0)
        st = load_training_state(path, model=m2, optimizer=opt2, scaler=None, map_location="cpu")
        assert st.next_epoch == 3
        assert st.global_step == 100
        torch.testing.assert_close(m.weight, m2.weight)
