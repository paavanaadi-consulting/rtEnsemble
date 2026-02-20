"""LSTM Ensemble Trading System - Core Package"""

from lstm_ensemble.complete_system import (
    Config,
    LSTMEnsemble,
    FeatureEngineering,
    LSTMModel,
    BiLSTMModel,
    AttentionLSTMModel,
    CNNLSTMModel,
    LiveDataReader,
    LiveInferenceEngine,
    PredictionAccuracyAnalyzer,
)

from lstm_ensemble.watchlist_features import (
    WATCHLIST_FEATURE_COLUMNS,
    extract_alert_features,
    build_watchlist_dataframe,
    build_sequences,
)

from lstm_ensemble.watchlist_ensemble import (
    WatchlistEnsemble,
    WatchlistLSTM,
    WatchlistBiLSTM,
    WatchlistAttention,
    WatchlistMLP,
)

__version__ = "1.0.0"

__all__ = [
    # Original OHLCV-based system
    "Config",
    "LSTMEnsemble",
    "FeatureEngineering",
    "LSTMModel",
    "BiLSTMModel",
    "AttentionLSTMModel",
    "CNNLSTMModel",
    "LiveDataReader",
    "LiveInferenceEngine",
    "PredictionAccuracyAnalyzer",
    # Watchlist alert-based system
    "WatchlistEnsemble",
    "WatchlistLSTM",
    "WatchlistBiLSTM",
    "WatchlistAttention",
    "WatchlistMLP",
    "WATCHLIST_FEATURE_COLUMNS",
    "extract_alert_features",
    "build_watchlist_dataframe",
    "build_sequences",
]
