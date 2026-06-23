"""Data loading and preprocessing for Netflix stock prediction."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def load_stock_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLCV CSV, parse dates, sort ascending, drop null OHLCV rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    before = len(df)
    df.dropna(subset=list(required), inplace=True)
    if dropped := before - len(df):
        logger.warning("Dropped %d rows with nulls", dropped)
    logger.info("Loaded %d trading days (%s → %s)", len(df), df.index[0].date(), df.index[-1].date())
    return df


def normalise(prices: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """Scale 1-D price array to [0, 1]. Returns (scaled_array, fitted_scaler)."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    return scaled, scaler


def create_sequences(prices: np.ndarray, window: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Slide fixed window over prices → supervised (X, y) arrays.

    Args:
        prices: 1-D normalised price array.
        window: Look-back window size.

    Returns:
        X of shape (n_samples, window), y of shape (n_samples,).

    Raises:
        ValueError: If prices is shorter than window + 1.
    """
    if len(prices) <= window:
        raise ValueError(f"prices length ({len(prices)}) must exceed window ({window})")
    X = np.array([prices[i - window:i] for i in range(window, len(prices))], dtype=np.float32)
    y = np.array([prices[i] for i in range(window, len(prices))], dtype=np.float32)
    return X, y


def build_feature_matrix(
    df: pd.DataFrame,
    window: int = 60,
    target_col: str = "Close",
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Normalise + sequence → (X, y, scaler) ready for Keras RNN."""
    prices = df[target_col].values
    _, scaler = normalise(prices)
    scaled = scaler.transform(prices.reshape(-1, 1)).flatten()
    X, y = create_sequences(scaled, window=window)
    X = X.reshape(X.shape[0], X.shape[1], 1)   # (samples, timesteps, 1)
    logger.info("Feature matrix: X=%s  y=%s", X.shape, y.shape)
    return X, y, scaler
