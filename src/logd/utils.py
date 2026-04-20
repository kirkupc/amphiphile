"""Shared utilities."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(level=os.environ.get("LOGD_LOG_LEVEL", "INFO"), format=LOG_FORMAT)
    return logger


def set_seed(seed: int) -> None:
    """Seed numpy + random + torch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def project_root() -> Path:
    """Repo root, resolved from this file's location."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    d = project_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    d = project_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def reports_dir() -> Path:
    d = project_root() / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d
