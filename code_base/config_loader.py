"""Load and deep-merge YAML/JSON training configs (base + experiment overlays)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import yaml

from .model import deep_merge_dict


def load_one_config_file(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config extension {path.suffix!r}; use .yaml, .yml, or .json")
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping at root: {path}")
    return data


def load_merged_config(paths: Sequence[Union[str, Path]]) -> Dict[str, Any]:
    """
    Load multiple configs and deep-merge left → right (later files override earlier).

    Example::

        cfg = load_merged_config(["configs/C1.yaml", "path/to/override.yaml"])
    """
    merged: Dict[str, Any] = {}
    for p in paths:
        merged = deep_merge_dict(merged, load_one_config_file(Path(p)))
    return merged
