"""Tests for the LSTM Ensemble Trading System."""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from lstm_ensemble.complete_system import (
    Config, LSTMEnsemble, FeatureEngineering,
    LSTMModel, BiLSTMModel, AttentionLSTMModel, CNNLSTMModel,
)


def make_synthetic_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)
    base = 150 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01 09:30", periods=n, freq="1min"),
        "open": base,
        "high": base + abs(np.random.randn(n) * 0.5),
        "low": base - abs(np.random.randn(n) * 0.5),
        "close": base + np.random.randn(n) * 0.2,
        "volume": np.random.randint(1000, 100000, n),
        "vwap": base,
        "transactions": np.random.randint(10, 500, n),
        "symbol": "TEST",
    })


class TestConfig:
    def test_default_config(self):
        config = Config()
        assert len(config.features) == 27
        assert config.sequence_length == 60
        assert config.prediction_horizon == 5

    def test_custom_config(self):
        custom = {"models": {"sequence_length": 30, "prediction_horizon": 10, "device": "cpu"}}
        config = Config(config_dict=custom)
        assert config.sequence_length == 30
        assert config.prediction_horizon == 10


class TestFeatureEngineering:
    def test_compute_features(self):
        df = make_synthetic_ohlcv(200)
        result = FeatureEngineering.compute_all_features(df)
        config = Config()

        # All feature columns should be present
        for feat in config.features:
            assert feat in result.columns, f"Missing feature: {feat}"

        # No NaN values in feature columns
        assert result[config.features].isna().sum().sum() == 0

    def test_no_infinities(self):
        df = make_synthetic_ohlcv(200)
        result = FeatureEngineering.compute_all_features(df)
        config = Config()
        for feat in config.features:
            assert not np.isinf(result[feat].values).any(), f"Inf in {feat}"

    def test_time_features_range(self):
        df = make_synthetic_ohlcv(200)
        result = FeatureEngineering.compute_all_features(df)
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()
        assert result["is_market_open"].isin([0.0, 1.0]).all()


class TestModels:
    def test_lstm_forward(self):
        model = LSTMModel(input_size=27, hidden_size=64)
        x = torch.randn(4, 60, 27)
        out = model(x)
        assert out.shape == (4,)

    def test_bilstm_forward(self):
        model = BiLSTMModel(input_size=27, hidden_size=64)
        x = torch.randn(4, 60, 27)
        out = model(x)
        assert out.shape == (4,)

    def test_attention_lstm_forward(self):
        model = AttentionLSTMModel(input_size=27, hidden_size=64)
        x = torch.randn(4, 60, 27)
        out = model(x)
        assert out.shape == (4,)

    def test_cnn_lstm_forward(self):
        model = CNNLSTMModel(input_size=27, hidden_size=64)
        x = torch.randn(4, 60, 27)
        out = model(x)
        assert out.shape == (4,)


class TestLSTMEnsemble:
    def _make_ensemble(self):
        config = Config()
        return LSTMEnsemble(
            feature_columns=config.features,
            sequence_length=60,
            device="cpu",
        )

    def test_prepare_sequences(self):
        df = make_synthetic_ohlcv(300)
        df = FeatureEngineering.compute_all_features(df)
        df["target"] = df["close"].pct_change(5).shift(-5)
        df = df.dropna()

        ensemble = self._make_ensemble()
        X, y = ensemble.prepare_sequences(df, "target")

        assert X.ndim == 3
        assert X.shape[1] == 60
        assert X.shape[2] == 27
        assert len(y) == len(X)

    def test_predict_individual(self):
        df = make_synthetic_ohlcv(300)
        df = FeatureEngineering.compute_all_features(df)
        df["target"] = df["close"].pct_change(5).shift(-5)
        df = df.dropna()

        ensemble = self._make_ensemble()
        X, y = ensemble.prepare_sequences(df, "target")

        preds = ensemble.predict(X[:8])
        assert "lstm" in preds
        assert "bilstm" in preds
        assert "attention_lstm" in preds
        assert "cnn_lstm" in preds

    def test_predict_ensemble(self):
        df = make_synthetic_ohlcv(300)
        df = FeatureEngineering.compute_all_features(df)
        df["target"] = df["close"].pct_change(5).shift(-5)
        df = df.dropna()

        ensemble = self._make_ensemble()
        X, y = ensemble.prepare_sequences(df, "target")

        ens_pred = ensemble.predict_ensemble(X[:8])
        assert isinstance(ens_pred, float)

    def test_save_load_roundtrip(self):
        df = make_synthetic_ohlcv(300)
        df = FeatureEngineering.compute_all_features(df)
        df["target"] = df["close"].pct_change(5).shift(-5)
        df = df.dropna()

        ensemble = self._make_ensemble()
        X, y = ensemble.prepare_sequences(df, "target")
        preds_before = ensemble.predict(X[:4])

        with tempfile.TemporaryDirectory() as tmpdir:
            ensemble.save_models(tmpdir)

            ensemble2 = self._make_ensemble()
            ensemble2.load_models(tmpdir)
            preds_after = ensemble2.predict(X[:4])

        for name in preds_before:
            assert abs(preds_before[name] - preds_after[name]) < 1e-5, \
                f"{name} prediction changed after load: {preds_before[name]} vs {preds_after[name]}"

    def test_flatten_for_trees(self):
        ensemble = self._make_ensemble()
        seq_data = np.random.randn(10, 60, 27).astype(np.float32)
        flat = ensemble._flatten_for_trees(seq_data)
        assert flat.shape == (10, 27 * 4)  # last, mean, std, diff
        assert not np.isnan(flat).any()
        assert not np.isinf(flat).any()

    def test_pytorch_training(self):
        """Test that PyTorch training runs without error (quick, 2 epochs)."""
        df = make_synthetic_ohlcv(300)
        df = FeatureEngineering.compute_all_features(df)
        df["target"] = df["close"].pct_change(5).shift(-5)
        df = df.dropna()

        ensemble = self._make_ensemble()
        X, y = ensemble.prepare_sequences(df, "target")

        split = int(len(X) * 0.8)
        ensemble.train_pytorch_models(
            X[:split], y[:split],
            X[split:], y[split:],
            epochs=2, batch_size=32,
        )

        # Models should still produce output after training
        preds = ensemble.predict(X[:4])
        assert len(preds) >= 4


if __name__ == "__main__":
    # Simple test runner
    passed = 0
    failed = 0
    errors = []

    for cls_name in [TestConfig, TestFeatureEngineering, TestModels, TestLSTMEnsemble]:
        obj = cls_name()
        for method_name in dir(obj):
            if method_name.startswith("test_"):
                try:
                    print(f"  {cls_name.__name__}.{method_name}...", end=" ")
                    getattr(obj, method_name)()
                    print("PASS")
                    passed += 1
                except Exception as e:
                    print(f"FAIL: {e}")
                    errors.append(f"{cls_name.__name__}.{method_name}: {e}")
                    failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for err in errors:
            print(f"  - {err}")
    print(f"{'='*50}")
    sys.exit(1 if failed else 0)
