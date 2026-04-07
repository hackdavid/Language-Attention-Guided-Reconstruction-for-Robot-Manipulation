"""Shared logging setup for training and data loading (debug-friendly step logs)."""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

DEFAULT_LOGGER_NAME = "la_reconvla"


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    return logging.getLogger(name)


def configure_training_logging(
    level: int = logging.INFO,
    *,
    logger_name: str = DEFAULT_LOGGER_NAME,
    format_str: str = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Attach a stderr handler to the root logger once so all ``code_base.*`` loggers emit.
    Idempotent if a StreamHandler is already on the root logger.
    """
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(format_str, datefmt=datefmt))
        root.addHandler(handler)
    root.setLevel(level)
    log = logging.getLogger(logger_name)
    log.setLevel(level)
    return log


def log_level_from_env(default: int = logging.INFO) -> int:
    v = os.environ.get("TRAINING_LOG_LEVEL", "").strip().upper()
    if v in ("DEBUG", "10"):
        return logging.DEBUG
    if v in ("INFO", "20"):
        return logging.INFO
    if v in ("WARNING", "WARN", "30"):
        return logging.WARNING
    if v in ("ERROR", "40"):
        return logging.ERROR
    return default
