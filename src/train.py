"""Training loop and evaluation utilities for stock price prediction."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, and R² for regression predictions.

    Args:
        y_true: Ground-truth values (1-D array).
        y_pred: Model predictions (1-D array, same length as y_true).

    Returns:
        Dictionary with keys "rmse", "mae", "r2".

    Raises:
        ValueError: If arrays have different lengths or are empty.
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true length {len(y_true)} != y_pred length {len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("Arrays must not be empty")

    residuals = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    logger.info("Metrics — RMSE: %.4f  MAE: %.4f  R²: %.4f", rmse, mae, r2)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_model(
    model: "keras.Model",
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    callbacks: list | None = None,
) -> "keras.callbacks.History":
    """Fit a Keras model and return the training history.

    Args:
        model: A compiled Keras model.
        X_train: Training features (samples, timesteps, features).
        y_train: Training targets (samples,).
        X_val: Validation features.
        y_val: Validation targets.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        callbacks: List of Keras callbacks (EarlyStopping etc.).

    Returns:
        Keras History object.
    """
    logger.info(
        "Training %s  |  train=%d  val=%d  epochs=%d  batch=%d",
        model.name,
        len(X_train),
        len(X_val),
        epochs,
        batch_size,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks or [],
        verbose=1,
    )
    return history


def train_test_split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Time-aware split that keeps chronological order (no shuffling).

    Args:
        X: Feature array (samples, timesteps, features).
        y: Target array (samples,).
        test_size: Fraction of data held out for testing.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    split = int(len(X) * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:]


def compare_models(
    models: Dict[str, "keras.Model"],
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: "MinMaxScaler",
) -> Dict[str, Dict[str, float]]:
    """Evaluate multiple trained models and return a metrics comparison dict.

    Args:
        models: Mapping of model name -> fitted Keras model.
        X_test: Test features.
        y_test: Normalised test targets.
        scaler: Fitted MinMaxScaler used to inverse-transform predictions.

    Returns:
        Nested dict: {model_name: {rmse, mae, r2}} in original price scale.
    """
    results = {}
    y_true_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    for name, model in models.items():
        y_pred_norm = model.predict(X_test, verbose=0).flatten()
        y_pred_orig = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        metrics = compute_metrics(y_true_orig, y_pred_orig)
        results[name] = metrics
        logger.info("[%s] RMSE=%.2f  MAE=%.2f  R²=%.4f", name, metrics["rmse"], metrics["mae"], metrics["r2"])

    return results
