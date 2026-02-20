"""
LSTM Ensemble Trading System - Complete Implementation
=======================================================
Contains all model architectures, feature engineering, training, and inference logic.
"""

import os
import pickle
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import yaml
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    """Central configuration manager."""

    _DEFAULT_CONFIG = {
        "database": {
            "connection_string": os.getenv("DB_CONNECTION", ""),
            "market_data_table": "market_data_minute",
            "predictions_table": "model_predictions",
            "signals_table": "trading_signals",
        },
        "polygon": {
            "api_key": os.getenv("POLYGON_API_KEY", ""),
            "feed": "delayed",
            "batch_size": 100,
        },
        "symbols": ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT"],
        "models": {
            "sequence_length": 60,
            "prediction_horizon": 5,
            "device": os.getenv("DEVICE", "cpu"),
        },
        "training": {
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 0.001,
            "validation_split": 0.2,
            "early_stopping_patience": 10,
        },
        "inference": {
            "interval_seconds": 60,
            "enable_signals": True,
            "min_confidence": 0.5,
        },
        "features": [
            "returns", "log_returns", "high_low_pct", "close_open_pct",
            "rsi", "macd", "macd_signal", "macd_hist",
            "bb_position", "bb_width",
            "price_to_sma5", "price_to_sma20", "ema_diff",
            "relative_volume", "volume_change",
            "price_to_vwap", "volume_momentum", "money_flow_ratio",
            "volatility_10", "volatility_20", "normalized_atr",
            "hour_sin", "hour_cos", "minute_sin", "minute_cos",
            "is_market_open", "is_morning",
        ],
        "ensemble_weights": {
            "lstm": 0.25,
            "bilstm": 0.25,
            "attention_lstm": 0.20,
            "cnn_lstm": 0.15,
            "lgbm": 0.10,
            "xgb": 0.05,
        },
    }

    def __init__(self, config_dict: dict = None, config_path: str = None):
        if config_dict:
            self._config = {**self._DEFAULT_CONFIG, **config_dict}
        elif config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self._config = {**self._DEFAULT_CONFIG, **yaml.safe_load(f)}
        else:
            # Try default path
            default_path = Path(__file__).parent.parent / "config" / "config.yaml"
            if default_path.exists():
                with open(default_path, "r") as f:
                    loaded = yaml.safe_load(f) or {}
                self._config = {**self._DEFAULT_CONFIG, **loaded}
            else:
                self._config = self._DEFAULT_CONFIG.copy()

        # Resolve env vars in connection string
        conn = self._config["database"]["connection_string"]
        if "${" in str(conn):
            self._config["database"]["connection_string"] = os.getenv(
                "DB_CONNECTION", conn
            )

        device_cfg = self._config["models"].get("device", "cpu")
        if "${" in str(device_cfg):
            device_cfg = os.getenv("DEVICE", "cpu")
        self._config["models"]["device"] = device_cfg

    # Convenience properties
    @property
    def features(self) -> List[str]:
        return self._config["features"]

    @property
    def sequence_length(self) -> int:
        return self._config["models"]["sequence_length"]

    @property
    def prediction_horizon(self) -> int:
        return self._config["models"]["prediction_horizon"]

    @property
    def device(self) -> str:
        return self._config["models"]["device"]

    @property
    def ensemble_weights(self) -> Dict[str, float]:
        return self._config["ensemble_weights"]


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

class FeatureEngineering:
    """Compute technical indicators from OHLCV data."""

    @staticmethod
    def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)

        # --- Price returns ---
        df["returns"] = close.pct_change()
        df["log_returns"] = np.log(close / close.shift(1))
        df["high_low_pct"] = (high - low) / close
        df["close_open_pct"] = (close - open_) / open_

        # --- RSI (14-period) ---
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14, min_periods=1).mean()
        avg_loss = loss.rolling(14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))
        # Normalize RSI to [-1, 1] range for better neural network input
        df["rsi"] = (df["rsi"] - 50) / 50

        # --- MACD ---
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = (ema12 - ema26) / close  # normalize by price
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # --- Bollinger Bands ---
        sma20 = close.rolling(20, min_periods=1).mean()
        std20 = close.rolling(20, min_periods=1).std()
        upper_band = sma20 + 2 * std20
        lower_band = sma20 - 2 * std20
        band_width = upper_band - lower_band
        df["bb_position"] = (close - lower_band) / (band_width + 1e-10)
        df["bb_width"] = band_width / sma20

        # --- Moving average features ---
        sma5 = close.rolling(5, min_periods=1).mean()
        df["price_to_sma5"] = (close - sma5) / (sma5 + 1e-10)
        df["price_to_sma20"] = (close - sma20) / (sma20 + 1e-10)

        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        df["ema_diff"] = (ema9 - ema21) / close

        # --- Volume indicators ---
        avg_vol = volume.rolling(20, min_periods=1).mean()
        df["relative_volume"] = volume / (avg_vol + 1e-10)
        df["volume_change"] = volume.pct_change()
        df["volume_momentum"] = volume.rolling(5, min_periods=1).mean() / (avg_vol + 1e-10)

        # --- VWAP ---
        if "vwap" in df.columns:
            vwap = df["vwap"].astype(float)
            df["price_to_vwap"] = (close - vwap) / (vwap + 1e-10)
        else:
            df["price_to_vwap"] = 0.0

        # --- Money flow ---
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        pos_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        neg_flow = money_flow.where(typical_price <= typical_price.shift(1), 0)
        pos_sum = pos_flow.rolling(14, min_periods=1).sum()
        neg_sum = neg_flow.rolling(14, min_periods=1).sum()
        df["money_flow_ratio"] = pos_sum / (neg_sum + 1e-10)
        # Cap to reasonable range
        df["money_flow_ratio"] = df["money_flow_ratio"].clip(-10, 10)

        # --- Volatility ---
        df["volatility_10"] = close.pct_change().rolling(10, min_periods=1).std()
        df["volatility_20"] = close.pct_change().rolling(20, min_periods=1).std()

        # --- ATR ---
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(14, min_periods=1).mean()
        df["normalized_atr"] = atr / close

        # --- Time features ---
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            hour = ts.dt.hour
            minute = ts.dt.minute
            df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            df["minute_sin"] = np.sin(2 * np.pi * minute / 60)
            df["minute_cos"] = np.cos(2 * np.pi * minute / 60)
            df["is_market_open"] = ((hour >= 9) & (hour < 16)).astype(float)
            df["is_morning"] = ((hour >= 9) & (hour < 12)).astype(float)
        else:
            for col in ["hour_sin", "hour_cos", "minute_sin", "minute_cos"]:
                df[col] = 0.0
            df["is_market_open"] = 1.0
            df["is_morning"] = 0.0

        # Replace infinities and fill NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()

        return df


# ---------------------------------------------------------------------------
# PyTorch Model Architectures
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """Vanilla LSTM with residual connection and layer normalization."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.layer_norm_in = nn.LayerNorm(hidden_size)

        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.layer_norm_in(x)
        out, _ = self.lstm(x)
        out = self.layer_norm(out[:, -1, :])
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM with gated residual output."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.layer_norm_in = nn.LayerNorm(hidden_size)

        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        # Gate for combining forward/backward
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.layer_norm_in(x)
        out, _ = self.lstm(x)
        out = self.layer_norm(out[:, -1, :])
        out = self.dropout(out)
        gate_val = self.gate(out)
        # Split forward/backward and gate
        fwd = out[:, :out.size(1) // 2]
        bwd = out[:, out.size(1) // 2:]
        gated = gate_val * fwd + (1 - gate_val) * bwd
        return self.fc(gated).squeeze(-1)


class AttentionLSTMModel(nn.Module):
    """LSTM with multi-head self-attention over temporal dimension."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.layer_norm_in = nn.LayerNorm(hidden_size)

        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads,
            dropout=dropout * 0.5,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.layer_norm_in(x)
        lstm_out, _ = self.lstm(x)

        # Self-attention over all time steps
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(attn_out + lstm_out)  # residual

        # Weighted pooling: use last hidden state as query
        last_hidden = attn_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out).squeeze(-1)


class CNNLSTMModel(nn.Module):
    """CNN for local pattern extraction followed by LSTM for temporal modeling."""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        # Multi-scale CNN
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_size, hidden_size // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
        )

        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features) -> Conv1d needs (batch, features, seq)
        c = x.permute(0, 2, 1)
        c = self.conv1(c)
        c = self.conv2(c)
        c = c.permute(0, 2, 1)  # back to (batch, seq, hidden)

        out, _ = self.lstm(c)
        out = self.layer_norm(out[:, -1, :])
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)


# ---------------------------------------------------------------------------
# Ensemble System
# ---------------------------------------------------------------------------

class LSTMEnsemble:
    """Complete ensemble system managing 4 LSTM + 2 tree models."""

    MODEL_NAMES = ["lstm", "bilstm", "attention_lstm", "cnn_lstm"]
    TREE_NAMES = ["lgbm", "xgb"]

    def __init__(self, feature_columns: List[str],
                 sequence_length: int = 60,
                 hidden_size: int = 128,
                 device: str = "cpu"):
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.device = torch.device(device if device != "${DEVICE:-cuda}" else "cpu")
        self.input_size = len(feature_columns)

        # Scaler
        self.scaler = StandardScaler()
        self._scaler_fitted = False

        # PyTorch models
        self.models: Dict[str, nn.Module] = {
            "lstm": LSTMModel(self.input_size, hidden_size),
            "bilstm": BiLSTMModel(self.input_size, hidden_size),
            "attention_lstm": AttentionLSTMModel(self.input_size, hidden_size),
            "cnn_lstm": CNNLSTMModel(self.input_size, hidden_size),
        }
        for m in self.models.values():
            m.to(self.device)

        # Tree models (initialized during training or loading)
        self.tree_models: Dict[str, Any] = {}

        # Default ensemble weights
        self.ensemble_weights = {
            "lstm": 0.25, "bilstm": 0.25,
            "attention_lstm": 0.20, "cnn_lstm": 0.15,
            "lgbm": 0.10, "xgb": 0.05,
        }

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_sequences(self, df: pd.DataFrame,
                          target_column: str = "target") -> Tuple[np.ndarray, np.ndarray]:
        """Create overlapping sequences from DataFrame.

        Returns:
            features: (N, seq_len, n_features)
            targets: (N,)
        """
        feature_values = df[self.feature_columns].values.astype(np.float32)
        targets = df[target_column].values.astype(np.float32)

        # Fit scaler if not already fitted
        if not self._scaler_fitted:
            self.scaler.fit(feature_values)
            self._scaler_fitted = True

        scaled = self.scaler.transform(feature_values)

        X, y = [], []
        for i in range(len(scaled) - self.sequence_length):
            X.append(scaled[i : i + self.sequence_length])
            y.append(targets[i + self.sequence_length])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_pytorch_models(self, train_data: np.ndarray, train_targets: np.ndarray,
                             val_data: np.ndarray, val_targets: np.ndarray,
                             epochs: int = 50, batch_size: int = 64,
                             learning_rate: float = 0.001):
        """Train all four PyTorch models with early stopping and LR scheduling."""

        train_X = torch.tensor(train_data, dtype=torch.float32).to(self.device)
        train_y = torch.tensor(train_targets, dtype=torch.float32).to(self.device)
        val_X = torch.tensor(val_data, dtype=torch.float32).to(self.device)
        val_y = torch.tensor(val_targets, dtype=torch.float32).to(self.device)

        train_dataset = TensorDataset(train_X, train_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.train()

            optimizer = optim.AdamW(
                model.parameters(), lr=learning_rate,
                weight_decay=1e-4,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.01,
            )
            criterion = nn.HuberLoss(delta=0.5)

            best_val_loss = float("inf")
            patience_counter = 0
            patience = 10
            best_state = None

            for epoch in range(epochs):
                # --- Train ---
                model.train()
                train_losses = []
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    pred = model(batch_X)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_losses.append(loss.item())

                scheduler.step()

                # --- Validate ---
                model.eval()
                with torch.no_grad():
                    val_pred = model(val_X)
                    val_loss = criterion(val_pred, val_y).item()

                avg_train = np.mean(train_losses)

                marker = ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    marker = " * [best]"
                else:
                    patience_counter += 1

                if (epoch + 1) % 5 == 0 or epoch == 0 or marker:
                    print(
                        f"  Epoch {epoch+1:3d}/{epochs}: "
                        f"train_loss={avg_train:.6f}  val_loss={val_loss:.6f}{marker}"
                    )

                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                    break

            # Restore best weights
            if best_state:
                model.load_state_dict(best_state)
            model.to(self.device)
            model.eval()
            print(f"  {name} best val_loss: {best_val_loss:.6f}")

    def train_tree_models(self, train_data: np.ndarray, train_targets: np.ndarray,
                          val_data: np.ndarray, val_targets: np.ndarray):
        """Train LightGBM and XGBoost on flattened sequence data."""

        # Flatten sequences: use last timestep + rolling stats
        train_flat = self._flatten_for_trees(train_data)
        val_flat = self._flatten_for_trees(val_data)

        # --- LightGBM ---
        try:
            import lightgbm as lgb
            print("\nTraining LightGBM...")
            lgb_train = lgb.Dataset(train_flat, label=train_targets)
            lgb_val = lgb.Dataset(val_flat, label=val_targets, reference=lgb_train)

            params = {
                "objective": "regression",
                "metric": "huber",
                "boosting_type": "gbdt",
                "num_leaves": 63,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_child_samples": 20,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "verbose": -1,
            }

            callbacks = [
                lgb.early_stopping(stopping_rounds=20, verbose=True),
                lgb.log_evaluation(period=50),
            ]

            self.tree_models["lgbm"] = lgb.train(
                params, lgb_train,
                num_boost_round=500,
                valid_sets=[lgb_val],
                callbacks=callbacks,
            )
            print(f"  LightGBM trained: {self.tree_models['lgbm'].best_iteration} rounds")
        except ImportError:
            print("  LightGBM not available, skipping")
        except Exception as e:
            print(f"  LightGBM training error: {e}")

        # --- XGBoost ---
        try:
            import xgboost as xgb
            print("\nTraining XGBoost...")
            dtrain = xgb.DMatrix(train_flat, label=train_targets)
            dval = xgb.DMatrix(val_flat, label=val_targets)

            params = {
                "objective": "reg:pseudohubererror",
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "eval_metric": "rmse",
                "verbosity": 0,
            }

            self.tree_models["xgb"] = xgb.train(
                params, dtrain,
                num_boost_round=500,
                evals=[(dval, "val")],
                early_stopping_rounds=20,
                verbose_eval=50,
            )
            print(f"  XGBoost trained: {self.tree_models['xgb'].best_iteration} rounds")
        except ImportError:
            print("  XGBoost not available, skipping")
        except Exception as e:
            print(f"  XGBoost training error: {e}")

    def _flatten_for_trees(self, seq_data: np.ndarray) -> np.ndarray:
        """Flatten sequence data for tree models using summary statistics."""
        data = np.asarray(seq_data, dtype=np.float64)
        last = data[:, -1, :]  # last timestep
        mean_vals = data.mean(axis=1)
        std_vals = data.std(axis=1)
        # First-last difference (trend)
        diff = data[:, -1, :] - data[:, 0, :]
        result = np.hstack([last, mean_vals, std_vals, diff])
        # Replace any remaining NaN/Inf
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, feature_data: np.ndarray) -> Dict[str, float]:
        """Get individual model predictions.

        Args:
            feature_data: Either raw features (n_samples, n_features) to be scaled,
                          or pre-scaled features (seq_len, n_features) / (1, seq_len, n_features).
        Returns:
            Dict mapping model name to prediction value.
        """
        seq = self._prepare_input(feature_data)
        results = {}

        # PyTorch models
        tensor_input = torch.tensor(seq, dtype=torch.float32).to(self.device)
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                pred = model(tensor_input)
            results[name] = float(pred.cpu().numpy().mean())

        # Tree models
        flat = self._flatten_for_trees(seq)
        for name, tree_model in self.tree_models.items():
            try:
                if name == "lgbm":
                    pred = tree_model.predict(flat)
                elif name == "xgb":
                    import xgboost as xgb
                    pred = tree_model.predict(xgb.DMatrix(flat))
                results[name] = float(np.mean(pred))
            except Exception:
                results[name] = 0.0

        return results

    def predict_ensemble(self, feature_data: np.ndarray) -> float:
        """Get weighted ensemble prediction."""
        individual = self.predict(feature_data)
        total_weight = 0.0
        weighted_sum = 0.0
        for name, pred in individual.items():
            w = self.ensemble_weights.get(name, 0.0)
            weighted_sum += w * pred
            total_weight += w
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0

    def _prepare_input(self, feature_data: np.ndarray) -> np.ndarray:
        """Normalize and reshape input for models."""
        data = np.array(feature_data, dtype=np.float32)

        # If 2D, check if it needs scaling
        if data.ndim == 2:
            if self._scaler_fitted:
                if data.shape[1] == self.input_size:
                    data = self.scaler.transform(data)
            # Trim to last sequence_length rows
            if data.shape[0] >= self.sequence_length:
                data = data[-self.sequence_length:]
            data = data[np.newaxis, :, :]  # (1, seq, features)
        elif data.ndim == 3:
            pass  # already (batch, seq, features)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {data.ndim}D")

        return data

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_models(self, model_dir: str):
        """Save all models, scaler, and config to directory."""
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        # PyTorch models
        for name, model in self.models.items():
            path = os.path.join(model_dir, f"{name}_model.pth")
            torch.save(model.state_dict(), path)

        # Tree models
        if "lgbm" in self.tree_models:
            path = os.path.join(model_dir, "lgbm_model.txt")
            self.tree_models["lgbm"].save_model(path)
            # Also save as pkl for compatibility
            with open(os.path.join(model_dir, "lgbm_model.pkl"), "wb") as f:
                pickle.dump(self.tree_models["lgbm"], f)

        if "xgb" in self.tree_models:
            path = os.path.join(model_dir, "xgb_model.json")
            self.tree_models["xgb"].save_model(path)
            with open(os.path.join(model_dir, "xgb_model.pkl"), "wb") as f:
                pickle.dump(self.tree_models["xgb"], f)

        # Scaler
        with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        # Metadata
        meta = {
            "feature_columns": self.feature_columns,
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "input_size": self.input_size,
            "ensemble_weights": self.ensemble_weights,
        }
        with open(os.path.join(model_dir, "model_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Models saved to {model_dir}")

    def load_models(self, model_dir: str):
        """Load all models from directory."""
        # Load metadata if available
        meta_path = os.path.join(model_dir, "model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if "ensemble_weights" in meta:
                self.ensemble_weights = meta["ensemble_weights"]

        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            self._scaler_fitted = True

        # Load PyTorch models
        for name in self.MODEL_NAMES:
            path = os.path.join(model_dir, f"{name}_model.pth")
            if os.path.exists(path):
                state = torch.load(path, map_location=self.device, weights_only=True)
                self.models[name].load_state_dict(state)
                self.models[name].to(self.device)
                self.models[name].eval()

        # Load tree models
        lgbm_path = os.path.join(model_dir, "lgbm_model.pkl")
        lgbm_txt = os.path.join(model_dir, "lgbm_model.txt")
        if os.path.exists(lgbm_path):
            with open(lgbm_path, "rb") as f:
                self.tree_models["lgbm"] = pickle.load(f)
        elif os.path.exists(lgbm_txt):
            try:
                import lightgbm as lgb
                self.tree_models["lgbm"] = lgb.Booster(model_file=lgbm_txt)
            except ImportError:
                pass

        xgb_path = os.path.join(model_dir, "xgb_model.pkl")
        xgb_json = os.path.join(model_dir, "xgb_model.json")
        if os.path.exists(xgb_path):
            with open(xgb_path, "rb") as f:
                self.tree_models["xgb"] = pickle.load(f)
        elif os.path.exists(xgb_json):
            try:
                import xgboost as xgb
                booster = xgb.Booster()
                booster.load_model(xgb_json)
                self.tree_models["xgb"] = booster
            except ImportError:
                pass

        print(f"Loaded models from {model_dir}")
        loaded = list(self.models.keys()) + list(self.tree_models.keys())
        print(f"  Active models: {', '.join(loaded)}")


# ---------------------------------------------------------------------------
# Live Data Reader
# ---------------------------------------------------------------------------

class LiveDataReader:
    """Read live market data from PostgreSQL for inference."""

    def __init__(self, db_connection_string: str, table_name: str = "market_data_minute"):
        from sqlalchemy import create_engine
        self.engine = create_engine(db_connection_string)
        self.table_name = table_name

    def get_latest_data(self, symbol: str, n_bars: int = 500) -> pd.DataFrame:
        from sqlalchemy import text
        query = text(f"""
            SELECT * FROM {self.table_name}
            WHERE symbol = :symbol
            ORDER BY timestamp DESC
            LIMIT :limit
        """)
        df = pd.read_sql(query, self.engine, params={"symbol": symbol, "limit": n_bars})
        if not df.empty:
            df = df.iloc[::-1].reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# Live Inference Engine
# ---------------------------------------------------------------------------

class LiveInferenceEngine:
    """Continuous live inference loop."""

    def __init__(self, ensemble: LSTMEnsemble, data_reader: LiveDataReader,
                 feature_columns: List[str],
                 db_connection_string: str = None):
        self.ensemble = ensemble
        self.data_reader = data_reader
        self.feature_columns = feature_columns
        self.db_connection_string = db_connection_string

    def run_single_prediction(self, symbol: str) -> Optional[Dict]:
        """Run a single prediction for a symbol."""
        df = self.data_reader.get_latest_data(symbol)
        if len(df) < self.ensemble.sequence_length + 50:
            return None

        df = FeatureEngineering.compute_all_features(df)
        df = df.dropna()

        if len(df) < self.ensemble.sequence_length:
            return None

        feature_data = df[self.feature_columns].values
        individual = self.ensemble.predict(feature_data)
        ensemble_pred = self.ensemble.predict_ensemble(feature_data)
        current_price = float(df["close"].iloc[-1])

        return {
            "symbol": symbol,
            "timestamp": df["timestamp"].iloc[-1],
            "current_price": current_price,
            "ensemble_pred": ensemble_pred,
            "individual_predictions": individual,
            "predicted_price": current_price * (1 + ensemble_pred),
        }

    def start_live_inference(self, symbol: str = "AAPL",
                             interval_seconds: int = 60):
        """Start continuous inference loop."""
        import time
        print(f"Starting live inference for {symbol} (interval: {interval_seconds}s)")
        try:
            while True:
                result = self.run_single_prediction(symbol)
                if result:
                    print(
                        f"[{result['timestamp']}] {symbol}: "
                        f"${result['current_price']:.2f} -> "
                        f"${result['predicted_price']:.2f} "
                        f"({result['ensemble_pred']*100:+.3f}%)"
                    )
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\nStopping live inference.")


# ---------------------------------------------------------------------------
# Prediction Accuracy Analyzer
# ---------------------------------------------------------------------------

class PredictionAccuracyAnalyzer:
    """Analyze prediction accuracy against actual prices."""

    def __init__(self, db_connection_string: str):
        from sqlalchemy import create_engine
        self.engine = create_engine(db_connection_string)

    def calculate_accuracy_metrics(self, symbol: str,
                                   lookback_hours: int = 24) -> Dict:
        from sqlalchemy import text
        query = text("""
            SELECT
                p.timestamp,
                p.ensemble_pred,
                p.prediction_confidence,
                p.model_agreement,
                m.close as actual_price
            FROM model_predictions p
            JOIN market_data_minute m
                ON p.symbol = m.symbol AND p.timestamp = m.timestamp
            WHERE p.symbol = :symbol
                AND p.timestamp >= NOW() - INTERVAL ':hours hours'
            ORDER BY p.timestamp ASC
        """)

        try:
            df = pd.read_sql(query, self.engine,
                             params={"symbol": symbol, "hours": lookback_hours})
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            return {
                "total_predictions": 0,
                "directional_accuracy": 0.0,
                "avg_confidence": 0.0,
                "avg_agreement": 0.0,
            }

        return {
            "total_predictions": len(df),
            "directional_accuracy": 0.0,
            "avg_confidence": float(df["prediction_confidence"].mean()),
            "avg_agreement": float(df["model_agreement"].mean()),
        }

    def print_accuracy_report(self, metrics: Dict, symbol: str):
        print(f"\nAccuracy Report for {symbol}")
        print("=" * 40)
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
