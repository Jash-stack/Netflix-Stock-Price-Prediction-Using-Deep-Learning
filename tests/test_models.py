"""Tests for the deep learning model module."""
import numpy as np
import pytest


@pytest.fixture
def sample_sequences():
    """50 sequences of length 50, each with 5 features."""
    np.random.seed(42)
    X = np.random.randn(50, 50, 5).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)
    return X, y


class TestDataPreprocessing:
    def test_create_sequences_correct_shape(self):
        from src.data_loader import create_sequences
        prices = np.arange(100, dtype=np.float32)
        X, y = create_sequences(prices, window=10)
        assert X.shape[1] == 10
        assert len(X) == len(y)
        assert len(X) == 90  # 100 - 10

    def test_normalised_values_in_range(self):
        from src.data_loader import normalise
        prices = np.array([100.0, 200.0, 150.0, 300.0])
        scaled, scaler = normalise(prices)
        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0

    def test_inverse_transform_recovers_original(self):
        from src.data_loader import normalise
        prices = np.array([100.0, 200.0, 150.0, 300.0])
        scaled, scaler = normalise(prices)
        recovered = scaler.inverse_transform(scaled.reshape(-1, 1)).flatten()
        np.testing.assert_allclose(recovered, prices, rtol=1e-5)


class TestModelArchitectures:
    def test_gru_output_shape(self, sample_sequences):
        from src.models import build_gru
        X, _ = sample_sequences
        model = build_gru(input_shape=(X.shape[1], X.shape[2]))
        preds = model.predict(X[:5], verbose=0)
        assert preds.shape == (5, 1)

    def test_lstm_output_shape(self, sample_sequences):
        from src.models import build_lstm
        X, _ = sample_sequences
        model = build_lstm(input_shape=(X.shape[1], X.shape[2]))
        preds = model.predict(X[:5], verbose=0)
        assert preds.shape == (5, 1)


class TestEvaluationMetrics:
    def test_rmse_non_negative(self):
        from src.train import compute_metrics
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 295.0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0
        assert 0 <= metrics["r2"] <= 1

    def test_perfect_prediction_r2_is_one(self):
        from src.train import compute_metrics
        y = np.array([100.0, 200.0, 300.0])
        metrics = compute_metrics(y, y)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)
