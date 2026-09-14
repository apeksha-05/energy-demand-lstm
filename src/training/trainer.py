"""
Shared training loop for all neural forecasting models in this project.
Handles batching, the forward/backward pass, validation, early stopping,
and history tracking — used identically by RNN, LSTM, Stacked LSTM,
BiLSTM, GRU, Attention-LSTM, and Transformer models.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE, RANDOM_SEED
from src.training.callbacks import EarlyStopping
from src.utils import get_logger, set_global_seed

logger = get_logger(__name__)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    patience: int = EARLY_STOPPING_PATIENCE,
) -> dict:
    """
    Train a PyTorch forecasting model with Adam, MSE loss, and early
    stopping on validation loss.

    Returns
    -------
    dict
        Training history: {"train_loss": [...], "val_loss": [...],
        "training_time_seconds": float, "num_parameters": int,
        "epochs_trained": int}
    """
    set_global_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Training %s | %d parameters | device=%s", model.__class__.__name__, num_parameters, device)

    history = {"train_loss": [], "val_loss": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_losses = []
        num_batches = len(train_loader)
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                logger.info("  batch %d/%d", batch_idx, num_batches)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_t)
            val_loss = loss_fn(val_predictions, y_val_t).item()

        train_loss = float(np.mean(epoch_train_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info(
            "Epoch %d/%d | train_loss=%.6f | val_loss=%.6f",
            epoch, epochs, train_loss, val_loss,
        )

        early_stopping.step(val_loss, model)
        if early_stopping.should_stop:
            logger.info("Early stopping triggered at epoch %d (best val_loss=%.6f)", epoch, early_stopping.best_loss)
            break

    early_stopping.restore_best(model)
    training_time = time.time() - start_time

    history["training_time_seconds"] = training_time
    history["num_parameters"] = num_parameters
    history["epochs_trained"] = epoch

    logger.info("Training complete: %.1fs, %d epochs, %d parameters", training_time, epoch, num_parameters)
    return history


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Run inference on a fitted model, returning a numpy array."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(device)
        predictions = model(X_t)
    return predictions.cpu().numpy()