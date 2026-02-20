"""
Watchlist Alert LSTM Ensemble
==============================
Ensemble model trained on sequences of watchlist alerts (last 3 alerts)
to predict small-cap price movements.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from lstm_ensemble.watchlist_features import (
    WATCHLIST_FEATURE_COLUMNS,
    extract_alert_features,
    build_watchlist_dataframe,
    build_sequences,
)


# ---------------------------------------------------------------------------
# Model Architectures (adapted for short 3-step sequences)
# ---------------------------------------------------------------------------

class WatchlistLSTM(nn.Module):
    """LSTM for short alert sequences (3 steps)."""

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.norm_in = nn.LayerNorm(hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.norm_out = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.norm_in(x)
        out, _ = self.lstm(x)
        out = self.norm_out(out[:, -1, :])
        out = self.dropout(out)
        return self.head(out).squeeze(-1)


class WatchlistBiLSTM(nn.Module):
    """Bidirectional LSTM for alert sequences."""

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.norm_in = nn.LayerNorm(hidden_size)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.norm_out = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.norm_in(x)
        out, _ = self.lstm(x)
        out = self.norm_out(out[:, -1, :])
        out = self.dropout(out)
        return self.head(out).squeeze(-1)


class WatchlistAttention(nn.Module):
    """Attention-based model for alert sequences.

    With only 3 time steps, attention is very effective at learning
    which of the 3 alerts matters most.
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.norm_in = nn.LayerNorm(hidden_size)

        # Positional embedding for 3 positions
        self.pos_emb = nn.Parameter(torch.randn(1, 3, hidden_size) * 0.02)

        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads,
            dropout=dropout * 0.5,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.norm_in(x)
        # Add positional embedding (handle seq_len <= 3)
        seq_len = x.size(1)
        x = x + self.pos_emb[:, :seq_len, :]
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.norm_attn(attn_out + x)
        # Use last position as output
        out = self.dropout(attn_out[:, -1, :])
        return self.head(out).squeeze(-1)


class WatchlistMLP(nn.Module):
    """MLP that flattens the 3 alerts into one vector.

    Good baseline for short sequences where temporal order
    matters less than the feature combination.
    """

    def __init__(self, input_size: int, seq_len: int = 3,
                 hidden_size: int = 128, dropout: float = 0.2):
        super().__init__()
        flat_size = input_size * seq_len
        self.net = nn.Sequential(
            nn.Linear(flat_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        x = x.flatten(1)  # (batch, seq*features)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Watchlist Ensemble
# ---------------------------------------------------------------------------

class WatchlistEnsemble:
    """Ensemble system for watchlist alert-based predictions."""

    MODEL_NAMES = ["lstm", "bilstm", "attention", "mlp"]

    def __init__(self, seq_len: int = 3, hidden_size: int = 64,
                 device: str = "cpu"):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.feature_columns = WATCHLIST_FEATURE_COLUMNS
        self.input_size = len(self.feature_columns)
        self.device = torch.device(device)

        self.scaler = StandardScaler()
        self._scaler_fitted = False

        self.models: Dict[str, nn.Module] = {
            "lstm": WatchlistLSTM(self.input_size, hidden_size),
            "bilstm": WatchlistBiLSTM(self.input_size, hidden_size),
            "attention": WatchlistAttention(self.input_size, hidden_size),
            "mlp": WatchlistMLP(self.input_size, seq_len, hidden_size * 2),
        }
        for m in self.models.values():
            m.to(self.device)

        self.tree_models: Dict[str, Any] = {}

        self.ensemble_weights = {
            "lstm": 0.25,
            "bilstm": 0.25,
            "attention": 0.30,  # attention gets more weight for 3-step sequences
            "mlp": 0.10,
            "lgbm": 0.05,
            "xgb": 0.05,
        }

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(self, X: np.ndarray, y: np.ndarray = None,
                     fit_scaler: bool = False) -> tuple:
        """Scale features and return tensors.

        Args:
            X: (N, seq_len, n_features)
            y: (N,) optional targets
            fit_scaler: whether to fit the scaler on this data
        """
        N, S, F = X.shape
        flat = X.reshape(-1, F)

        if fit_scaler and not self._scaler_fitted:
            self.scaler.fit(flat)
            self._scaler_fitted = True

        if self._scaler_fitted:
            flat = self.scaler.transform(flat)

        X_scaled = flat.reshape(N, S, F).astype(np.float32)

        if y is not None:
            return X_scaled, y.astype(np.float32)
        return X_scaled

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_all(self, X: np.ndarray, y: np.ndarray,
                  val_split: float = 0.2,
                  epochs: int = 100, batch_size: int = 128,
                  learning_rate: float = 0.001):
        """Train all models on watchlist alert sequences."""

        # Scale data
        X_scaled, y_clean = self.prepare_data(X, y, fit_scaler=True)

        # Split
        n_val = int(len(X_scaled) * val_split)
        # Use last N for validation (temporal split)
        train_X, val_X = X_scaled[:-n_val], X_scaled[-n_val:]
        train_y, val_y = y_clean[:-n_val], y_clean[-n_val:]

        print(f"Training data: {train_X.shape[0]} samples")
        print(f"Validation data: {val_X.shape[0]} samples")
        print(f"Features per alert: {self.input_size}")
        print(f"Sequence length: {self.seq_len}")

        # Train PyTorch models
        self._train_pytorch(train_X, train_y, val_X, val_y,
                            epochs, batch_size, learning_rate)

        # Train tree models
        self._train_trees(train_X, train_y, val_X, val_y)

        # Print final validation metrics
        print(f"\n{'='*50}")
        print("VALIDATION RESULTS")
        print(f"{'='*50}")
        val_preds = self.predict_batch(val_X)
        self._print_metrics(val_y, val_preds)

    def _train_pytorch(self, train_X, train_y, val_X, val_y,
                       epochs, batch_size, learning_rate):
        """Train PyTorch models."""
        train_tensor_X = torch.tensor(train_X).to(self.device)
        train_tensor_y = torch.tensor(train_y).to(self.device)
        val_tensor_X = torch.tensor(val_X).to(self.device)
        val_tensor_y = torch.tensor(val_y).to(self.device)

        dataset = TensorDataset(train_tensor_X, train_tensor_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.train()

            optimizer = optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=1e-3,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=15, T_mult=2, eta_min=learning_rate * 0.01,
            )
            criterion = nn.HuberLoss(delta=0.02)  # tighter delta for small returns

            best_val_loss = float("inf")
            patience_counter = 0
            patience = 15
            best_state = None

            for epoch in range(epochs):
                model.train()
                losses = []
                for bx, by in loader:
                    optimizer.zero_grad()
                    pred = model(bx)
                    loss = criterion(pred, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(loss.item())

                scheduler.step()

                model.eval()
                with torch.no_grad():
                    val_pred = model(val_tensor_X)
                    val_loss = criterion(val_pred, val_tensor_y).item()

                avg_train = np.mean(losses)
                marker = ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    marker = " * [best]"
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0 or epoch == 0 or marker:
                    print(f"  Epoch {epoch+1:3d}/{epochs}: "
                          f"train={avg_train:.6f} val={val_loss:.6f}{marker}")

                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

            if best_state:
                model.load_state_dict(best_state)
            model.to(self.device)
            model.eval()

            # Print directional accuracy
            with torch.no_grad():
                vp = model(val_tensor_X).cpu().numpy()
            dir_acc = np.mean(np.sign(vp) == np.sign(val_y)) * 100
            print(f"  {name} val_loss={best_val_loss:.6f} dir_acc={dir_acc:.1f}%")

    def _train_trees(self, train_X, train_y, val_X, val_y):
        """Train LightGBM and XGBoost on flattened alert features."""
        train_flat = self._flatten(train_X)
        val_flat = self._flatten(val_X)

        # LightGBM
        try:
            import lightgbm as lgb
            print("\nTraining LightGBM...")
            dtrain = lgb.Dataset(train_flat, label=train_y)
            dval = lgb.Dataset(val_flat, label=val_y, reference=dtrain)

            params = {
                "objective": "regression",
                "metric": "huber",
                "boosting_type": "gbdt",
                "num_leaves": 31,
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
                lgb.early_stopping(stopping_rounds=30, verbose=True),
                lgb.log_evaluation(period=100),
            ]
            self.tree_models["lgbm"] = lgb.train(
                params, dtrain, num_boost_round=500,
                valid_sets=[dval], callbacks=callbacks,
            )
            print(f"  LightGBM best iter: {self.tree_models['lgbm'].best_iteration}")
        except ImportError:
            print("  LightGBM not installed, skipping")
        except Exception as e:
            print(f"  LightGBM error: {e}")

        # XGBoost
        try:
            import xgboost as xgb
            print("\nTraining XGBoost...")
            dtrain = xgb.DMatrix(train_flat, label=train_y)
            dval = xgb.DMatrix(val_flat, label=val_y)

            params = {
                "objective": "reg:pseudohubererror",
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "eval_metric": "rmse",
                "verbosity": 0,
            }
            self.tree_models["xgb"] = xgb.train(
                params, dtrain, num_boost_round=500,
                evals=[(dval, "val")],
                early_stopping_rounds=30,
                verbose_eval=100,
            )
            print(f"  XGBoost best iter: {self.tree_models['xgb'].best_iteration}")
        except ImportError:
            print("  XGBoost not installed, skipping")
        except Exception as e:
            print(f"  XGBoost error: {e}")

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """Flatten sequences for tree models."""
        # Concatenate all timesteps + add cross-step features
        data = np.asarray(X, dtype=np.float64)
        flat = data.reshape(data.shape[0], -1)  # (N, seq*features)
        # Add diff between last and first alert
        diff = data[:, -1, :] - data[:, 0, :]
        result = np.hstack([flat, diff])
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    def _print_metrics(self, y_true, y_pred_dict):
        """Print validation metrics for ensemble."""
        # Ensemble prediction
        ens_pred = np.zeros_like(y_true)
        total_w = 0
        for name, preds in y_pred_dict.items():
            w = self.ensemble_weights.get(name, 0)
            ens_pred += w * preds
            total_w += w
        if total_w > 0:
            ens_pred /= total_w

        dir_acc = np.mean(np.sign(ens_pred) == np.sign(y_true)) * 100
        mae = np.mean(np.abs(ens_pred - y_true))
        rmse = np.sqrt(np.mean((ens_pred - y_true) ** 2))

        # Win rate for predicted buys
        buy_mask = ens_pred > 0.002
        if buy_mask.any():
            buy_win = np.mean(y_true[buy_mask] > 0) * 100
            buy_avg_return = np.mean(y_true[buy_mask]) * 100
        else:
            buy_win = 0
            buy_avg_return = 0

        print(f"  Ensemble dir accuracy: {dir_acc:.1f}%")
        print(f"  MAE: {mae:.4f}  RMSE: {rmse:.4f}")
        print(f"  Buy signal win rate: {buy_win:.1f}% ({buy_mask.sum()} signals)")
        print(f"  Avg return on buys: {buy_avg_return:+.2f}%")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_batch(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from all models for a batch."""
        results = {}
        tensor_X = torch.tensor(X, dtype=torch.float32).to(self.device)

        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                preds = model(tensor_X).cpu().numpy()
            results[name] = preds

        flat = self._flatten(X)
        for name, tree_model in self.tree_models.items():
            try:
                if name == "lgbm":
                    results[name] = tree_model.predict(flat)
                elif name == "xgb":
                    import xgboost as xgb
                    results[name] = tree_model.predict(xgb.DMatrix(flat))
            except Exception:
                results[name] = np.zeros(len(X))

        return results

    def predict_single(self, alert_sequence: np.ndarray) -> Dict:
        """Predict for a single sequence of alerts.

        Args:
            alert_sequence: (seq_len, n_features) or (1, seq_len, n_features)

        Returns:
            Dict with ensemble_pred, individual predictions, confidence, signal
        """
        if alert_sequence.ndim == 2:
            alert_sequence = alert_sequence[np.newaxis, :, :]

        # Scale
        X = self.prepare_data(alert_sequence)
        if isinstance(X, tuple):
            X = X[0]

        preds = self.predict_batch(X)

        # Weighted ensemble
        ens = 0.0
        total_w = 0.0
        for name, p in preds.items():
            w = self.ensemble_weights.get(name, 0)
            ens += w * float(p[0])
            total_w += w
        if total_w > 0:
            ens /= total_w

        # Confidence
        pred_vals = [float(p[0]) for p in preds.values()]
        agreement = self._calc_agreement(pred_vals)
        magnitude = min(abs(ens) * 50, 1.0)  # scale for small-cap returns
        confidence = 0.6 * agreement + 0.4 * magnitude

        # Signal
        signal = "HOLD"
        strength = 0.0
        if confidence >= 0.4:
            if ens > 0.002:
                signal = "BUY"
                strength = min(ens * confidence * 20, 1.0)
            elif ens < -0.002:
                signal = "SELL"
                strength = min(abs(ens) * confidence * 20, 1.0)

        return {
            "ensemble_pred": ens,
            "individual_preds": {k: float(v[0]) for k, v in preds.items()},
            "confidence": confidence,
            "agreement": agreement,
            "signal": signal,
            "signal_strength": strength,
        }

    def _calc_agreement(self, preds: list) -> float:
        if len(preds) < 2:
            return 1.0
        mean_p = np.mean(preds)
        std_p = np.std(preds)
        if abs(mean_p) < 1e-10:
            return 1.0
        return max(0, 1 - std_p / (abs(mean_p) + 1e-10))

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_models(self, model_dir: str):
        """Save all models and scaler."""
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            torch.save(model.state_dict(),
                       os.path.join(model_dir, f"wl_{name}_model.pth"))

        if "lgbm" in self.tree_models:
            self.tree_models["lgbm"].save_model(
                os.path.join(model_dir, "wl_lgbm_model.txt"))
            with open(os.path.join(model_dir, "wl_lgbm_model.pkl"), "wb") as f:
                pickle.dump(self.tree_models["lgbm"], f)

        if "xgb" in self.tree_models:
            self.tree_models["xgb"].save_model(
                os.path.join(model_dir, "wl_xgb_model.json"))
            with open(os.path.join(model_dir, "wl_xgb_model.pkl"), "wb") as f:
                pickle.dump(self.tree_models["xgb"], f)

        with open(os.path.join(model_dir, "wl_scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)

        meta = {
            "feature_columns": self.feature_columns,
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "input_size": self.input_size,
            "ensemble_weights": self.ensemble_weights,
        }
        with open(os.path.join(model_dir, "wl_model_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Watchlist models saved to {model_dir}")

    def load_models(self, model_dir: str):
        """Load all models from directory."""
        meta_path = os.path.join(model_dir, "wl_model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if "ensemble_weights" in meta:
                self.ensemble_weights = meta["ensemble_weights"]

        scaler_path = os.path.join(model_dir, "wl_scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            self._scaler_fitted = True

        for name in self.MODEL_NAMES:
            path = os.path.join(model_dir, f"wl_{name}_model.pth")
            if os.path.exists(path):
                state = torch.load(path, map_location=self.device, weights_only=True)
                self.models[name].load_state_dict(state)
                self.models[name].to(self.device)
                self.models[name].eval()

        lgbm_pkl = os.path.join(model_dir, "wl_lgbm_model.pkl")
        if os.path.exists(lgbm_pkl):
            with open(lgbm_pkl, "rb") as f:
                self.tree_models["lgbm"] = pickle.load(f)

        xgb_pkl = os.path.join(model_dir, "wl_xgb_model.pkl")
        if os.path.exists(xgb_pkl):
            with open(xgb_pkl, "rb") as f:
                self.tree_models["xgb"] = pickle.load(f)

        loaded = list(self.models.keys()) + list(self.tree_models.keys())
        print(f"Loaded watchlist models from {model_dir}: {', '.join(loaded)}")
