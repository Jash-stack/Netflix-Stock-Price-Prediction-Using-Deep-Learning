"""Deep-learning model builders for Netflix stock price prediction.

Exposes build_gru, build_lstm, build_rnn, and get_callbacks so that
training scripts and test suites can import them without coupling to a
specific architecture.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_rnn_base(
    cell_type: str,
    input_shape: Tuple[int, int],
    units: int = 64,
    dropout: float = 0.2,
    recurrent_dropout: float = 0.0,
) -> keras.Model:
    """Shared 2-layer RNN builder.

    Parameters
    ----------
    cell_type : str
        One of "GRU", "LSTM", or "SimpleRNN".
    input_shape : tuple
        (timesteps, features) tuple.
    units : int
        Number of hidden units per recurrent layer.
    dropout : float
        Fraction of units to drop on input gates.
    recurrent_dropout : float
        Fraction of units to drop on recurrent connections.

    Returns
    -------
    keras.Model
        Compiled model (Adam, MSE loss).
    """
    cell_map = {
        "GRU": layers.GRU,
        "LSTM": layers.LSTM,
        "SimpleRNN": layers.SimpleRNN,
    }
    if cell_type not in cell_map:
        raise ValueError(f"Unknown cell_type '{cell_type}'. Choose from {list(cell_map)}")

    RNNCell = cell_map[cell_type]

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            RNNCell(
                units,
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
            ),
            RNNCell(
                units // 2,
                return_sequences=False,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
            ),
            layers.Dense(32, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1),
        ],
        name=f"{cell_type.lower()}_predictor",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    logger.info("Built %s model: %d -> %d units", cell_type, units, units // 2)
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_gru(
    input_shape: Tuple[int, int],
    units: int = 64,
    dropout: float = 0.2,
) -> keras.Model:
    """Return a compiled 2-layer GRU model.

    Parameters
    ----------
    input_shape : tuple
        (timesteps, features) e.g. (60, 1).
    units : int
        Hidden units in the first GRU layer (second uses units // 2).
    dropout : float
        Dropout fraction applied to input and output connections.
    """
    return _build_rnn_base("GRU", input_shape, units=units, dropout=dropout)


def build_lstm(
    input_shape: Tuple[int, int],
    units: int = 64,
    dropout: float = 0.2,
) -> keras.Model:
    """Return a compiled 2-layer LSTM model.

    Parameters
    ----------
    input_shape : tuple
        (timesteps, features) e.g. (60, 1).
    units : int
        Hidden units in the first LSTM layer (second uses units // 2).
    dropout : float
        Dropout fraction applied to input and output connections.
    """
    return _build_rnn_base("LSTM", input_shape, units=units, dropout=dropout)


def build_rnn(
    input_shape: Tuple[int, int],
    units: int = 64,
    dropout: float = 0.2,
) -> keras.Model:
    """Return a compiled 2-layer vanilla SimpleRNN model.

    Parameters
    ----------
    input_shape : tuple
        (timesteps, features) e.g. (60, 1).
    units : int
        Hidden units in the first layer (second uses units // 2).
    dropout : float
        Dropout fraction applied to input and output connections.
    """
    return _build_rnn_base("SimpleRNN", input_shape, units=units, dropout=dropout)


def get_callbacks(
    patience: int = 10,
    model_path: str | Path = "best_model.keras",
) -> list:
    """Return a standard callback list for model training.

    Includes EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau.

    Parameters
    ----------
    patience : int
        Epochs with no improvement before early stopping / LR reduction.
    model_path : str or Path
        File path to save the best model checkpoint.
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    logger.info("Created %d training callbacks (patience=%d)", len(callbacks), patience)
    return callbacks
