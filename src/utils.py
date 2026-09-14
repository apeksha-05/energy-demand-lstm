"""
Shared utilities: logging configuration and reproducibility helpers.
"""

import logging
import random
import sys

import numpy as np


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger that writes to stdout with a consistent format.
    Safe to call multiple times for the same name (won't duplicate handlers).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_global_seed(seed: int) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch for reproducibility.
    Call this once at the start of any training or experiment script.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass