"""
Training callbacks: early stopping and best-model checkpointing.
Framework-agnostic (works with any PyTorch model + optimizer).
"""

import copy

import torch
import torch.nn as nn


class EarlyStopping:
    """
    Stops training when validation loss hasn't improved for `patience`
    consecutive epochs, and keeps a copy of the best-performing model's
    weights so training can restore them at the end.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state_dict: dict | None = None
        self.should_stop = False

    def step(self, val_loss: float, model: nn.Module) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model: nn.Module) -> None:
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)