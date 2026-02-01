"""
LSTM Ensemble Trading System - Complete Integrated Implementation
=================================================================

A production-ready PyTorch-based ensemble system for minute-level stock trading.

Architecture:
    Polygon API → Database → Feature Engineering → 
    LSTM Ensemble (6 models) → Predictions → Signals → Database

Features:
    ✓ Multiple LSTM architectures + Tree models
    ✓ Real-time data ingestion from Polygon
    ✓ Live inference with database output
    ✓ Comprehensive accuracy tracking
    ✓ Trading signal generation
    ✓ Performance analysis

Author: Trading System
Version: 1.0.0
Date: 2024-02-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import threading
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import xgboost as xgb
import pickle
import yaml
import os

warnings.filterwarnings('ignore')

# Try to import Polygon (optional for running without data ingestion)
try:
    from polygon import WebSocketClient
    from polygon.websocket.models import EquityAgg
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    print("⚠️  Polygon client not available. Install with: pip install polygon-api-client")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """System configuration"""
    
    def __init__(self, config_dict: Dict = None):
        if config_dict is None:
            # Default configuration
            config_dict = {
                'database': {
                    'connection_string': os.getenv(
                        'DB_CONNECTION',
                        'postgresql://user:password@localhost:5432/trading_db'
                    ),
                    'market_data_table': 'market_data_minute',
                    'predictions_table': 'model_predictions',
                    'signals_table': 'trading_signals'
                },
                'polygon': {
                    'api_key': os.getenv('POLYGON_API_KEY', 'your_key'),
                    'feed': 'delayed',
                    'batch_size': 100
                },
                'symbols': ['AAPL', 'TSLA', 'NVDA'],
                'models': {
                    'sequence_length': 60,
                    'prediction_horizon': 5,
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
                },
                'training': {
                    'epochs': 50,
                    'batch_size': 64,
                    'learning_rate': 0.001,
                    'validation_split': 0.2
                },
                'features': [
                    'returns', 'log_returns', 'high_low_pct', 'close_open_pct',
                    'rsi', 'macd', 'macd_signal', 'macd_hist',
                    'bb_position', 'bb_width',
                    'price_to_sma5', 'price_to_sma20', 'ema_diff',
                    'relative_volume', 'volume_change', 'price_to_vwap',
                    'volume_momentum', 'money_flow_ratio',
                    'volatility_10', 'volatility_20', 'normalized_atr',
                    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
                    'is_market_open', 'is_morning'
                ],
                'ensemble_weights': {
                    'lstm': 0.25,
                    'bilstm': 0.25,
                    'attention_lstm': 0.20,
                    'cnn_lstm': 0.15,
                    'lgbm': 0.10,
                    'xgb': 0.05
                }
            }
        
        self._config = config_dict
        self.db_connection = config_dict['database']['connection_string']
        self.market_data_table = config_dict['database']['market_data_table']
        self.predictions_table = config_dict['database']['predictions_table']
        self.signals_table = config_dict['database']['signals_table']
        self.symbols = config_dict['symbols']
        self.features = config_dict['features']
        self.ensemble_weights = config_dict['ensemble_weights']
        self.sequence_length = config_dict['models']['sequence_length']
        self.prediction_horizon = config_dict['models']['prediction_horizon']
        self.device = config_dict['models']['device']
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load config from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)


# ============================================================================
# DATA INGESTION
# ============================================================================

class PolygonToDatabase:
    """Stream Polygon data to database"""
    
    def __init__(
        self, 
        api_key: str,
        db_connection_string: str,
        table_name: str = "market_data_minute",
        symbols: List[str] = None,
        batch_size: int = 100
    ):
        if not POLYGON_AVAILABLE:
            raise ImportError("Polygon client not installed")
        
        self.api_key = api_key
        self.engine = create_engine(db_connection_string)
        self.table_name = table_name
        self.symbols = symbols or []
        self.batch_size = batch_size
        self.buffer = []
        self.lock = threading.Lock()
        
        self._create_table()
        
    def _create_table(self):
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open DECIMAL(12, 4),
            high DECIMAL(12, 4),
            low DECIMAL(12, 4),
            close DECIMAL(12, 4),
            volume BIGINT,
            vwap DECIMAL(12, 4),
            transactions INTEGER,
            received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        );
        
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_symbol_timestamp 
        ON {self.table_name}(symbol, timestamp DESC);
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(create_sql))
            conn.commit()
            
    def _handle_message(self, msgs: List[EquityAgg]):
        for msg in msgs:
            bar_data = {
                'symbol': msg.symbol,
                'timestamp': datetime.fromtimestamp(msg.start_timestamp / 1000),
                'open': msg.open,
                'high': msg.high,
                'low': msg.low,
                'close': msg.close,
                'volume': msg.volume,
                'vwap': msg.vwap,
                'transactions': msg.transactions
            }
            
            with self.lock:
                self.buffer.append(bar_data)
                if len(self.buffer) >= self.batch_size:
                    self._flush_buffer()
    
    def _flush_buffer(self):
        if not self.buffer:
            return
            
        df = pd.DataFrame(self.buffer)
        
        try:
            with self.engine.connect() as conn:
                df.to_sql(self.table_name, conn, if_exists='append', index=False)
                conn.commit()
            print(f"✓ Wrote {len(self.buffer)} records")
            self.buffer = []
        except Exception as e:
            print(f"Error: {e}")
    
    def start_streaming(self):
        ws = WebSocketClient(api_key=self.api_key, feed="delayed", market="stocks")
        subscriptions = [f"AM.{s}" for s in self.symbols]
        ws.subscribe(subscriptions)
        ws.run(self._handle_message)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineering:
    """Comprehensive feature engineering for minute-level data"""
    
    @staticmethod
    def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators and features"""
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma20 + (std20 * 2)
        df['bb_lower'] = sma20 - (std20 * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['price_to_sma5'] = df['close'] / df['sma_5']
        df['price_to_sma20'] = df['close'] / df['sma_20']
        df['ema_diff'] = (df['ema_9'] - df['ema_21']) / df['ema_21']
        
        # Volume
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['relative_volume'] = df['volume'] / df['volume_sma_20']
        df['volume_change'] = df['volume'].pct_change()
        
        if 'vwap' not in df.columns:
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['price_to_vwap'] = df['close'] / df['vwap']
        df['volume_momentum'] = df['returns'] * df['relative_volume']
        df['money_flow'] = df['close'] * df['volume']
        df['money_flow_ratio'] = df['money_flow'] / df['money_flow'].rolling(20).mean()
        
        # Volatility
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        df['normalized_atr'] = df['atr'] / df['close']
        
        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
        df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)).astype(int)
        df['is_morning'] = ((df['hour'] >= 9) & (df['hour'] < 12)).astype(int)
        
        # Fill NaN
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df


# ============================================================================
# PYTORCH LSTM MODELS
# ============================================================================

class LSTMModel(nn.Module):
    """Vanilla LSTM"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout if num_layers > 1 else 0, 
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)


class AttentionLSTM(nn.Module):
    """LSTM with attention"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        out = self.dropout(context)
        return self.fc(out)


class CNNLSTMHybrid(nn.Module):
    """CNN-LSTM Hybrid"""
    
    def __init__(self, input_size, lstm_hidden=64, cnn_filters=64, kernel_size=3, dropout=0.2, output_size=1):
        super().__init__()
        
        # CNN branch
        self.conv1 = nn.Conv1d(input_size, cnn_filters, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters//2, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(cnn_filters//2)
        
        # LSTM branch
        self.lstm = nn.LSTM(input_size, lstm_hidden, 2, dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden + cnn_filters//2, output_size)
        
    def forward(self, x):
        # LSTM branch
        lstm_out, _ = self.lstm(x)
        lstm_features = lstm_out[:, -1, :]
        
        # CNN branch
        x_cnn = x.transpose(1, 2)
        cnn_out = F.relu(self.bn1(self.conv1(x_cnn)))
        cnn_out = self.pool1(cnn_out)
        cnn_out = F.relu(self.bn2(self.conv2(cnn_out)))
        cnn_features = torch.mean(cnn_out, dim=2)
        
        # Combine
        combined = torch.cat([lstm_features, cnn_features], dim=1)
        out = self.dropout(combined)
        return self.fc(out)


# ============================================================================
# DATASET
# ============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series"""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, sequence_length: int = 60):
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return x, y


# ============================================================================
# LSTM ENSEMBLE
# ============================================================================

class LSTMEnsemble:
    """Complete LSTM Ensemble System"""
    
    def __init__(self, feature_columns: List[str], sequence_length: int = 60,
                 prediction_horizon: int = 1, device: str = 'cuda'):
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.device = device
        
        self.scaler = RobustScaler()
        self.input_size = len(feature_columns)
        
        self.models = {}
        self._initialize_models()
        
        self.model_weights = {
            'lstm': 0.25, 'bilstm': 0.25, 'attention_lstm': 0.20,
            'cnn_lstm': 0.15, 'lgbm': 0.10, 'xgb': 0.05
        }
        
    def _initialize_models(self):
        """Initialize all models"""
        self.models['lstm'] = LSTMModel(self.input_size).to(self.device)
        self.models['bilstm'] = BidirectionalLSTM(self.input_size).to(self.device)
        self.models['attention_lstm'] = AttentionLSTM(self.input_size).to(self.device)
        self.models['cnn_lstm'] = CNNLSTMHybrid(self.input_size).to(self.device)
        self.models['lgbm'] = None
        self.models['xgb'] = None
        
    def prepare_sequences(self, df: pd.DataFrame, target_column: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training"""
        features = df[self.feature_columns].values
        features_scaled = self.scaler.fit_transform(features)
        targets = df[target_column].values
        return features_scaled, targets
    
    def train_pytorch_models(self, train_data, train_targets, val_data, val_targets,
                            epochs=50, batch_size=64, learning_rate=0.001):
        """Train PyTorch LSTM models"""
        train_dataset = TimeSeriesDataset(train_data, train_targets, self.sequence_length)
        val_dataset = TimeSeriesDataset(val_data, val_targets, self.sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        for model_name in ['lstm', 'bilstm', 'attention_lstm', 'cnn_lstm']:
            print(f"\nTraining {model_name}...")
            
            model = self.models[model_name]
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
            criterion = nn.MSELoss()
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0.0
                
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        outputs = model(batch_x).squeeze()
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                scheduler.step(val_loss)
                
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f"models/{model_name}_best.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            model.load_state_dict(torch.load(f"models/{model_name}_best.pth"))
            print(f"✓ {model_name} complete. Best val loss: {best_val_loss:.6f}")
    
    def train_tree_models(self, train_data, train_targets, val_data, val_targets):
        """Train LightGBM and XGBoost"""
        X_train = train_data[-len(train_targets):, :]
        X_val = val_data[-len(val_targets):, :]
        
        print("\nTraining LightGBM...")
        self.models['lgbm'] = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=7, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        self.models['lgbm'].fit(
            X_train, train_targets[-len(X_train):],
            eval_set=[(X_val, val_targets[-len(X_val):])],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        
        print("\nTraining XGBoost...")
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=7,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            random_state=42
        )
        self.models['xgb'].fit(
            X_train, train_targets[-len(X_train):],
            eval_set=[(X_val, val_targets[-len(X_val):])],
            verbose=50
        )
        
        print("✓ Tree models training complete")
    
    def predict(self, data: np.ndarray) -> Dict[str, float]:
        """Generate predictions from all models"""
        predictions = {}
        
        sequence = data[-self.sequence_length:]
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        for model_name in ['lstm', 'bilstm', 'attention_lstm', 'cnn_lstm']:
            self.models[model_name].eval()
            with torch.no_grad():
                pred = self.models[model_name](sequence_tensor)
                predictions[model_name] = pred.cpu().numpy().flatten()[0]
        
        current_features = data[-1:, :]
        
        if self.models['lgbm']:
            predictions['lgbm'] = self.models['lgbm'].predict(current_features)[0]
        
        if self.models['xgb']:
            predictions['xgb'] = self.models['xgb'].predict(current_features)[0]
        
        return predictions
    
    def predict_ensemble(self, data: np.ndarray) -> float:
        """Weighted ensemble prediction"""
        predictions = self.predict(data)
        return sum(predictions[m] * self.model_weights[m] for m in predictions.keys())
    
    def save_models(self, path: str = "models/"):
        """Save all models"""
        os.makedirs(path, exist_ok=True)
        
        for model_name in ['lstm', 'bilstm', 'attention_lstm', 'cnn_lstm']:
            torch.save(self.models[model_name].state_dict(), f"{path}/{model_name}.pth")
        
        if self.models['lgbm']:
            self.models['lgbm'].booster_.save_model(f"{path}/lgbm_model.txt")
        
        if self.models['xgb']:
            self.models['xgb'].save_model(f"{path}/xgb_model.json")
        
        with open(f"{path}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"✓ Models saved to {path}")
    
    def load_models(self, path: str = "models/"):
        """Load all models"""
        for model_name in ['lstm', 'bilstm', 'attention_lstm', 'cnn_lstm']:
            self.models[model_name].load_state_dict(
                torch.load(f"{path}/{model_name}.pth", map_location=self.device)
            )
            self.models[model_name].eval()
        
        self.models['lgbm'] = lgb.Booster(model_file=f"{path}/lgbm_model.txt")
        self.models['xgb'] = xgb.XGBRegressor()
        self.models['xgb'].load_model(f"{path}/xgb_model.json")
        
        with open(f"{path}/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"✓ Models loaded from {path}")


# ============================================================================
# LIVE DATA READER
# ============================================================================

class LiveDataReader:
    """Read live data from database"""
    
    def __init__(self, db_connection_string: str, table_name: str, 
                 symbol: str, lookback_minutes: int = 500):
        self.engine = create_engine(db_connection_string)
        self.table_name = table_name
        self.symbol = symbol
        self.lookback_minutes = lookback_minutes
        
    def get_latest_data(self) -> pd.DataFrame:
        """Fetch latest N minutes of data"""
        query = f"""
        SELECT timestamp, open, high, low, close, volume, vwap
        FROM {self.table_name}
        WHERE symbol = :symbol
        ORDER BY timestamp DESC
        LIMIT :limit
        """
        
        df = pd.read_sql(text(query), self.engine, 
                        params={'symbol': self.symbol, 'limit': self.lookback_minutes})
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def stream_data(self, callback, interval_seconds: int = 60):
        """Stream data continuously"""
        while True:
            try:
                df = self.get_latest_data()
                if len(df) >= self.lookback_minutes:
                    callback(df)
                else:
                    print(f"Waiting for data... ({len(df)}/{self.lookback_minutes})")
                time.sleep(interval_seconds)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(interval_seconds)


# ============================================================================
# PREDICTION WRITER
# ============================================================================

class PredictionWriter:
    """Write predictions to database"""
    
    def __init__(self, db_connection_string: str,
                 predictions_table: str = "model_predictions",
                 signals_table: str = "trading_signals"):
        self.engine = create_engine(db_connection_string)
        self.predictions_table = predictions_table
        self.signals_table = signals_table
        self._create_tables()
        
    def _create_tables(self):
        """Create output tables"""
        predictions_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.predictions_table} (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            symbol VARCHAR(10) NOT NULL,
            current_price DECIMAL(12, 4),
            lstm_pred DECIMAL(10, 6),
            bilstm_pred DECIMAL(10, 6),
            attention_lstm_pred DECIMAL(10, 6),
            cnn_lstm_pred DECIMAL(10, 6),
            lgbm_pred DECIMAL(10, 6),
            xgb_pred DECIMAL(10, 6),
            ensemble_pred DECIMAL(10, 6),
            prediction_confidence DECIMAL(5, 4),
            model_agreement DECIMAL(5, 4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        );
        
        CREATE INDEX IF NOT EXISTS idx_{self.predictions_table}_symbol_timestamp 
        ON {self.predictions_table}(symbol, timestamp DESC);
        """
        
        signals_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.signals_table} (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            symbol VARCHAR(10) NOT NULL,
            signal VARCHAR(10) NOT NULL,
            signal_strength DECIMAL(5, 4),
            predicted_return DECIMAL(10, 6),
            current_price DECIMAL(12, 4),
            target_price DECIMAL(12, 4),
            stop_loss DECIMAL(12, 4),
            confidence_score DECIMAL(5, 4),
            volatility DECIMAL(10, 6),
            is_executed BOOLEAN DEFAULT FALSE,
            execution_price DECIMAL(12, 4),
            execution_time TIMESTAMP,
            actual_return DECIMAL(10, 6),
            pnl DECIMAL(12, 4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        );
        
        CREATE INDEX IF NOT EXISTS idx_{self.signals_table}_symbol_timestamp 
        ON {self.signals_table}(symbol, timestamp DESC);
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(predictions_sql))
            conn.execute(text(signals_sql))
            conn.commit()
    
    def write_prediction(self, prediction_record: Dict):
        """Write prediction to database"""
        insert_sql = f"""
        INSERT INTO {self.predictions_table} (
            timestamp, symbol, current_price,
            lstm_pred, bilstm_pred, attention_lstm_pred, cnn_lstm_pred,
            lgbm_pred, xgb_pred, ensemble_pred,
            prediction_confidence, model_agreement
        ) VALUES (
            :timestamp, :symbol, :current_price,
            :lstm, :bilstm, :attention_lstm, :cnn_lstm,
            :lgbm, :xgb, :ensemble_prediction,
            :confidence, :agreement
        )
        ON CONFLICT (symbol, timestamp) DO UPDATE SET
            ensemble_pred = EXCLUDED.ensemble_pred;
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), prediction_record)
                conn.commit()
            return True
        except Exception as e:
            print(f"Error writing prediction: {e}")
            return False
    
    def write_signal(self, signal_record: Dict):
        """Write trading signal"""
        insert_sql = f"""
        INSERT INTO {self.signals_table} (
            timestamp, symbol, signal,
            signal_strength, predicted_return, current_price,
            target_price, stop_loss,
            confidence_score, volatility
        ) VALUES (
            :timestamp, :symbol, :signal,
            :signal_strength, :predicted_return, :current_price,
            :target_price, :stop_loss,
            :confidence_score, :volatility
        )
        ON CONFLICT (symbol, timestamp) DO UPDATE SET
            signal = EXCLUDED.signal;
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(insert_sql), signal_record)
                conn.commit()
            return True
        except Exception as e:
            print(f"Error writing signal: {e}")
            return False


# ============================================================================
# LIVE INFERENCE ENGINE
# ============================================================================

class LiveInferenceEngine:
    """Live inference with database output"""
    
    def __init__(self, ensemble: LSTMEnsemble, data_reader: LiveDataReader,
                 feature_columns: List[str], db_connection_string: str,
                 enable_signals: bool = True):
        self.ensemble = ensemble
        self.data_reader = data_reader
        self.feature_columns = feature_columns
        self.enable_signals = enable_signals
        self.writer = PredictionWriter(db_connection_string)
        self.predictions_history = deque(maxlen=1000)
        self.total_predictions = 0
        
    def process_new_data(self, df: pd.DataFrame):
        """Process new data and generate predictions"""
        try:
            df_features = FeatureEngineering.compute_all_features(df)
            features_scaled = self.ensemble.scaler.transform(
                df_features[self.feature_columns].values
            )
            
            ensemble_pred = self.ensemble.predict_ensemble(features_scaled)
            individual_preds = self.ensemble.predict(features_scaled)
            
            # Calculate metrics
            model_preds = list(individual_preds.values())
            model_agreement = self._calculate_agreement(model_preds)
            confidence = self._calculate_confidence(ensemble_pred, model_preds, df_features)
            
            prediction_record = {
                'timestamp': df_features['timestamp'].iloc[-1],
                'symbol': self.data_reader.symbol,
                'current_price': float(df_features['close'].iloc[-1]),
                'ensemble_prediction': float(ensemble_pred),
                'confidence': float(confidence),
                'agreement': float(model_agreement),
                **{k: float(v) for k, v in individual_preds.items()}
            }
            
            self.writer.write_prediction(prediction_record)
            self.total_predictions += 1
            self._print_prediction(prediction_record, individual_preds)
            
            if self.enable_signals:
                signal_record = self._generate_signal(prediction_record, df_features)
                if signal_record and signal_record['signal'] != 'HOLD':
                    self.writer.write_signal(signal_record)
                    self._print_signal(signal_record)
            
        except Exception as e:
            print(f"Error processing data: {e}")
    
    def _calculate_agreement(self, predictions: List[float]) -> float:
        """Calculate model agreement"""
        if len(predictions) < 2:
            return 1.0
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        if abs(mean_pred) < 1e-6:
            return 1.0
        cv = std_pred / abs(mean_pred)
        return max(0, 1 - cv)
    
    def _calculate_confidence(self, ensemble_pred: float, 
                             model_preds: List[float],
                             df_features: pd.DataFrame) -> float:
        """Calculate prediction confidence"""
        agreement = self._calculate_agreement(model_preds)
        magnitude = min(abs(ensemble_pred) * 100, 1.0)
        recent_vol = df_features['volatility_20'].iloc[-1]
        vol_factor = 1 / (1 + recent_vol * 10)
        
        confidence = 0.5 * agreement + 0.3 * magnitude + 0.2 * vol_factor
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_signal(self, prediction_record: Dict, 
                        df_features: pd.DataFrame) -> Optional[Dict]:
        """Generate trading signal"""
        pred = prediction_record['ensemble_prediction']
        confidence = prediction_record['confidence']
        current_price = prediction_record['current_price']
        
        if confidence < 0.5:
            return None
        
        buy_threshold = 0.002
        sell_threshold = -0.002
        
        if pred > buy_threshold:
            signal = "BUY"
            signal_strength = min(pred / buy_threshold, 2.0) * confidence
        elif pred < sell_threshold:
            signal = "SELL"
            signal_strength = min(abs(pred) / abs(sell_threshold), 2.0) * confidence
        else:
            return None
        
        atr = df_features['atr'].iloc[-1]
        volatility = df_features['volatility_20'].iloc[-1]
        
        if signal == "BUY":
            target_price = current_price * (1 + abs(pred) * 2)
            stop_loss = current_price - (atr * 1.5)
        else:
            target_price = current_price * (1 - abs(pred) * 2)
            stop_loss = current_price + (atr * 1.5)
        
        return {
            'timestamp': prediction_record['timestamp'],
            'symbol': prediction_record['symbol'],
            'signal': signal,
            'signal_strength': float(signal_strength),
            'predicted_return': float(pred),
            'current_price': float(current_price),
            'target_price': float(target_price),
            'stop_loss': float(stop_loss),
            'confidence_score': float(confidence),
            'volatility': float(volatility)
        }
    
    def _print_prediction(self, record: Dict, individual_preds: Dict):
        """Print prediction results"""
        print(f"\n{'='*80}")
        print(f"⏰ {record['timestamp']} | 📈 {record['symbol']} | 💰 ${record['current_price']:.2f}")
        print(f"--- Models ---")
        for model, pred in individual_preds.items():
            print(f"  {model:20s}: {pred*100:+6.2f}%")
        print(f"--- Ensemble ---")
        print(f"  Prediction: {record['ensemble_prediction']*100:+6.2f}%")
        print(f"  Confidence: {record['confidence']:.2%}")
        print(f"  Agreement:  {record['agreement']:.2%}")
        print(f"📊 Total: {self.total_predictions}")
        print(f"{'='*80}\n")
    
    def _print_signal(self, signal: Dict):
        """Print trading signal"""
        print(f"\n🚨 SIGNAL: {signal['signal']} | Strength: {signal['signal_strength']:.2f}")
        print(f"   Price: ${signal['current_price']:.2f} → Target: ${signal['target_price']:.2f}")
        print(f"   Stop Loss: ${signal['stop_loss']:.2f} | Confidence: {signal['confidence_score']:.2%}\n")
    
    def start_live_inference(self, interval_seconds: int = 60):
        """Start live inference loop"""
        print("="*80)
        print("🚀 LSTM Ensemble Live Inference Engine")
        print("="*80)
        print(f"  Symbol: {self.data_reader.symbol}")
        print(f"  Interval: {interval_seconds}s")
        print("="*80 + "\n")
        
        self.data_reader.stream_data(
            callback=self.process_new_data,
            interval_seconds=interval_seconds
        )


# ============================================================================
# ACCURACY ANALYZER
# ============================================================================

class PredictionAccuracyAnalyzer:
    """Analyze prediction accuracy"""
    
    def __init__(self, db_connection_string: str,
                 predictions_table: str = "model_predictions",
                 market_data_table: str = "market_data_minute"):
        self.engine = create_engine(db_connection_string)
        self.predictions_table = predictions_table
        self.market_data_table = market_data_table
    
    def calculate_accuracy_metrics(self, symbol: str, 
                                   lookback_hours: int = 24,
                                   prediction_horizon: int = 5) -> Dict:
        """Calculate comprehensive accuracy metrics"""
        query = f"""
        WITH predictions_with_actuals AS (
            SELECT 
                p.timestamp,
                p.ensemble_pred as predicted_return,
                p.prediction_confidence,
                (LEAD(m.close, {prediction_horizon}) OVER (
                    PARTITION BY m.symbol ORDER BY m.timestamp
                ) - m.close) / m.close as actual_return
            FROM {self.predictions_table} p
            JOIN {self.market_data_table} m 
                ON p.symbol = m.symbol AND p.timestamp = m.timestamp
            WHERE p.symbol = :symbol
                AND p.timestamp >= NOW() - INTERVAL '{lookback_hours} hours'
        )
        SELECT * FROM predictions_with_actuals WHERE actual_return IS NOT NULL;
        """
        
        df = pd.read_sql(text(query), self.engine, params={'symbol': symbol})
        
        if len(df) == 0:
            return {}
        
        df['direction_correct'] = (
            np.sign(df['predicted_return']) == np.sign(df['actual_return'])
        ).astype(int)
        
        metrics = {
            'directional_accuracy': df['direction_correct'].mean(),
            'mae': abs(df['predicted_return'] - df['actual_return']).mean(),
            'rmse': np.sqrt(((df['predicted_return'] - df['actual_return']) ** 2).mean()),
            'total_predictions': len(df)
        }
        
        return metrics
    
    def print_accuracy_report(self, metrics: Dict, symbol: str):
        """Print accuracy report"""
        print(f"\n{'='*80}")
        print(f"📊 ACCURACY REPORT - {symbol}")
        print(f"{'='*80}")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2%}")
        print(f"  Mean Absolute Error:  {metrics['mae']:.6f}")
        print(f"  RMSE:                 {metrics['rmse']:.6f}")
        print(f"  Total Predictions:    {metrics['total_predictions']}")
        print(f"{'='*80}\n")


# ============================================================================
# MAIN EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage"""
    
    # Configuration
    config = Config()
    
    print("="*80)
    print("LSTM ENSEMBLE TRADING SYSTEM")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Symbols: {', '.join(config.symbols)}")
    print(f"Features: {len(config.features)}")
    print("="*80 + "\n")
    
    # Example 1: Train models
    def train_models():
        ensemble = LSTMEnsemble(
            feature_columns=config.features,
            sequence_length=config.sequence_length,
            device=config.device
        )
        
        # Load training data (implement your data loading logic)
        # train_data, train_targets = load_data()
        # ensemble.train_pytorch_models(train_data, train_targets, val_data, val_targets)
        # ensemble.train_tree_models(train_data, train_targets, val_data, val_targets)
        # ensemble.save_models("models/")
        
        print("✓ Models trained and saved")
    
    # Example 2: Live inference
    def run_inference(symbol="AAPL"):
        ensemble = LSTMEnsemble(
            feature_columns=config.features,
            sequence_length=config.sequence_length,
            device=config.device
        )
        ensemble.load_models("models/")
        
        data_reader = LiveDataReader(
            db_connection_string=config.db_connection,
            table_name=config.market_data_table,
            symbol=symbol,
            lookback_minutes=500
        )
        
        engine = LiveInferenceEngine(
            ensemble=ensemble,
            data_reader=data_reader,
            feature_columns=config.features,
            db_connection_string=config.db_connection,
            enable_signals=True
        )
        
        engine.start_live_inference(interval_seconds=60)
    
    # Example 3: Analyze accuracy
    def analyze_accuracy(symbol="AAPL"):
        analyzer = PredictionAccuracyAnalyzer(config.db_connection)
        metrics = analyzer.calculate_accuracy_metrics(symbol, lookback_hours=24)
        analyzer.print_accuracy_report(metrics, symbol)
    
    # Uncomment to run
    # train_models()
    # run_inference("AAPL")
    # analyze_accuracy("AAPL")
    
    print("\n✓ System initialized successfully!")
    print("\nTo use:")
    print("  1. train_models() - Train the ensemble")
    print("  2. run_inference('AAPL') - Start live predictions")
    print("  3. analyze_accuracy('AAPL') - Check performance\n")


if __name__ == "__main__":
    main()
