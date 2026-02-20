#!/usr/bin/env python3
"""
Watchlist Alerts LSTM Ensemble Inference
=========================================

Reads latest records from public.watchlist_alerts, performs predictions
using the LSTM ensemble, and writes results to public.lstm_ensemble_output.

This script continuously monitors the watchlist_alerts table for new entries
and generates ensemble predictions for each alert.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import yaml
from dotenv import load_dotenv

import sqlalchemy as sa
from sqlalchemy import create_engine, text
import torch

from lstm_ensemble.complete_system import (
    Config, LSTMEnsemble, FeatureEngineering
)


# Load environment variables
load_dotenv()


class WatchlistDataReader:
    """Reads data from watchlist_alerts table"""
    
    def __init__(self, db_connection_string: str):
        self.engine = create_engine(db_connection_string)
        self.last_processed_id = self._get_last_processed_id()
        
    def _get_last_processed_id(self) -> int:
        """Get the last processed alert ID"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT COALESCE(MAX(watchlist_alert_id), 0) FROM public.lstm_ensemble_output"
                ))
                return result.scalar() or 0
        except Exception as e:
            print(f"Warning: Could not get last processed ID (table may not exist yet): {e}")
            return 0
    
    def get_new_alerts(self) -> pd.DataFrame:
        """Fetch new alerts from watchlist_alerts table"""
        try:
            query = text("""
                SELECT 
                    id as alert_id,
                    symbol,
                    timestamp
                FROM public.watchlist_alerts
                WHERE id > :last_id
                ORDER BY id ASC
                LIMIT 100
            """)
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={'last_id': self.last_processed_id})
            
            if not df.empty:
                print(f"📥 Found {len(df)} new alerts (IDs: {df['alert_id'].min()} - {df['alert_id'].max()})")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching alerts: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, symbol: str, timestamp: datetime, lookback_minutes: int = 500) -> pd.DataFrame:
        """Get historical market data for feature engineering"""
        try:
            start_time = timestamp - timedelta(minutes=lookback_minutes)
            
            query = text("""
                SELECT
                    symbol,
                    timestamp,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    vwap,
                    transactions
                FROM public.market_data_minute
                WHERE symbol = :symbol
                    AND timestamp >= :start_time
                    AND timestamp <= :end_time
                ORDER BY timestamp ASC
            """)
            
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    query, 
                    conn, 
                    params={
                        'symbol': symbol,
                        'start_time': start_time,
                        'end_time': timestamp
                    }
                )
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def update_last_processed_id(self, alert_id: int):
        """Update the last processed alert ID"""
        if alert_id > self.last_processed_id:
            self.last_processed_id = alert_id


class EnsembleOutputWriter:
    """Writes ensemble predictions to lstm_ensemble_output table"""
    
    def __init__(self, db_connection_string: str):
        self.engine = create_engine(db_connection_string)
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Create the output table if it doesn't exist"""
        create_table_sql = text("""
            CREATE TABLE IF NOT EXISTS public.lstm_ensemble_output (
                symbol VARCHAR(10) PRIMARY KEY,
                watchlist_alert_id BIGINT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                alert_type VARCHAR(50),
                current_price DECIMAL(12, 4),
                
                -- Individual model predictions
                lstm_pred DECIMAL(10, 6),
                bilstm_pred DECIMAL(10, 6),
                attention_lstm_pred DECIMAL(10, 6),
                cnn_lstm_pred DECIMAL(10, 6),
                lgbm_pred DECIMAL(10, 6),
                xgb_pred DECIMAL(10, 6),
                
                -- Ensemble prediction
                ensemble_pred DECIMAL(10, 6),
                predicted_return_pct DECIMAL(10, 4),
                predicted_price DECIMAL(12, 4),
                
                -- Confidence metrics
                prediction_confidence DECIMAL(5, 4),
                model_agreement DECIMAL(5, 4),
                
                -- Trading signal
                signal VARCHAR(10),
                signal_strength DECIMAL(5, 4),
                target_price DECIMAL(12, 4),
                stop_loss DECIMAL(12, 4),
                
                -- Metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_lstm_output_signal 
                ON public.lstm_ensemble_output(signal);
            CREATE INDEX IF NOT EXISTS idx_lstm_output_updated 
                ON public.lstm_ensemble_output(updated_at DESC);
        """)
        
        try:
            with self.engine.connect() as conn:
                conn.execute(create_table_sql)
                conn.commit()
            print("✅ Output table ready: public.lstm_ensemble_output")
        except Exception as e:
            print(f"⚠️  Table creation warning: {e}")
    
    def write_prediction(self, prediction: Dict) -> bool:
        """Write prediction to database"""
        try:
            insert_sql = text("""
                INSERT INTO public.lstm_ensemble_output (
                    symbol, watchlist_alert_id, timestamp, alert_type, current_price,
                    lstm_pred, bilstm_pred, attention_lstm_pred, cnn_lstm_pred, 
                    lgbm_pred, xgb_pred, ensemble_pred, 
                    predicted_return_pct, predicted_price,
                    prediction_confidence, model_agreement,
                    signal, signal_strength, target_price, stop_loss,
                    updated_at
                ) VALUES (
                    :symbol, :watchlist_alert_id, :timestamp, :alert_type, :current_price,
                    :lstm_pred, :bilstm_pred, :attention_lstm_pred, :cnn_lstm_pred,
                    :lgbm_pred, :xgb_pred, :ensemble_pred,
                    :predicted_return_pct, :predicted_price,
                    :prediction_confidence, :model_agreement,
                    :signal, :signal_strength, :target_price, :stop_loss,
                    CURRENT_TIMESTAMP
                )
                ON CONFLICT (symbol) DO UPDATE SET
                    watchlist_alert_id = EXCLUDED.watchlist_alert_id,
                    timestamp = EXCLUDED.timestamp,
                    alert_type = EXCLUDED.alert_type,
                    current_price = EXCLUDED.current_price,
                    lstm_pred = EXCLUDED.lstm_pred,
                    bilstm_pred = EXCLUDED.bilstm_pred,
                    attention_lstm_pred = EXCLUDED.attention_lstm_pred,
                    cnn_lstm_pred = EXCLUDED.cnn_lstm_pred,
                    lgbm_pred = EXCLUDED.lgbm_pred,
                    xgb_pred = EXCLUDED.xgb_pred,
                    ensemble_pred = EXCLUDED.ensemble_pred,
                    predicted_return_pct = EXCLUDED.predicted_return_pct,
                    predicted_price = EXCLUDED.predicted_price,
                    prediction_confidence = EXCLUDED.prediction_confidence,
                    model_agreement = EXCLUDED.model_agreement,
                    signal = EXCLUDED.signal,
                    signal_strength = EXCLUDED.signal_strength,
                    target_price = EXCLUDED.target_price,
                    stop_loss = EXCLUDED.stop_loss,
                    updated_at = CURRENT_TIMESTAMP
            """)
            
            with self.engine.connect() as conn:
                conn.execute(insert_sql, prediction)
                conn.commit()
            
            return True
            
        except Exception as e:
            print(f"❌ Error writing prediction: {e}")
            return False


class WatchlistInferenceEngine:
    """Main inference engine for watchlist alerts"""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            self.config = Config(config_dict)
        else:
            self.config = Config()
        
        # Database connection
        db_connection = os.getenv('DB_CONNECTION')
        if not db_connection:
            raise ValueError("DB_CONNECTION environment variable not set!")
        
        # Initialize components
        self.data_reader = WatchlistDataReader(db_connection)
        self.output_writer = EnsembleOutputWriter(db_connection)
        
        # Load ensemble models
        self.ensembles = {}  # symbol -> ensemble mapping
        
        print("✅ Watchlist Inference Engine initialized")
    
    def load_ensemble_for_symbol(self, symbol: str) -> Optional[LSTMEnsemble]:
        """Load trained ensemble for a symbol"""
        if symbol in self.ensembles:
            return self.ensembles[symbol]
        
        model_path = Path(f"models/{symbol}")
        if not model_path.exists():
            print(f"⚠️  No trained models found for {symbol} at {model_path}")
            return None
        
        try:
            ensemble = LSTMEnsemble(
                feature_columns=self.config.features,
                sequence_length=self.config.sequence_length,
                device=self.config.device
            )
            ensemble.load_models(str(model_path))
            self.ensembles[symbol] = ensemble
            print(f"✅ Loaded ensemble for {symbol}")
            return ensemble
            
        except Exception as e:
            print(f"❌ Error loading ensemble for {symbol}: {e}")
            return None
    
    def process_alert(self, alert: pd.Series) -> Optional[Dict]:
        """Process a single watchlist alert"""
        symbol = alert['symbol']
        alert_id = alert['alert_id']
        timestamp = pd.to_datetime(alert['timestamp'])
        
        print(f"\n{'='*60}")
        print(f"Processing Alert #{alert_id}: {symbol} @ {timestamp}")
        
        # Load ensemble
        ensemble = self.load_ensemble_for_symbol(symbol)
        if ensemble is None:
            print(f"⚠️  Skipping {symbol} - no trained model")
            return None
        
        # Get historical data for features
        historical_df = self.data_reader.get_historical_data(
            symbol, 
            timestamp,
            lookback_minutes=500
        )
        
        if len(historical_df) < self.config.sequence_length:
            print(f"⚠️  Insufficient historical data for {symbol} ({len(historical_df)} bars)")
            return None
        
        try:
            # Compute features
            df_features = FeatureEngineering.compute_all_features(historical_df)
            df_features = df_features.dropna()

            # Get current price from historical data (most recent close)
            current_price = float(historical_df['close'].iloc[-1])
            print(f"Current Price: ${current_price:.2f}")

            # Get raw feature values - the ensemble.predict handles scaling
            feature_values = df_features[self.config.features].values

            # Get predictions (predict handles scaling internally)
            ensemble_pred = ensemble.predict_ensemble(feature_values)
            individual_preds = ensemble.predict(feature_values)
            
            # Calculate metrics
            model_preds = list(individual_preds.values())
            model_agreement = self._calculate_agreement(model_preds)
            confidence = self._calculate_confidence(ensemble_pred, model_preds, df_features)
            
            # Calculate predicted price
            predicted_return_pct = ensemble_pred * 100  # Convert to percentage
            predicted_price = current_price * (1 + ensemble_pred)
            
            # Generate signal
            signal_info = self._generate_signal(
                ensemble_pred, 
                confidence, 
                current_price,
                df_features
            )
            
            # Create prediction record
            prediction = {
                'watchlist_alert_id': int(alert_id),
                'symbol': symbol,
                'timestamp': timestamp,
                'alert_type': None,  # Not available in simplified alert query
                'current_price': current_price,
                'lstm_pred': float(individual_preds.get('lstm', 0)),
                'bilstm_pred': float(individual_preds.get('bilstm', 0)),
                'attention_lstm_pred': float(individual_preds.get('attention_lstm', 0)),
                'cnn_lstm_pred': float(individual_preds.get('cnn_lstm', 0)),
                'lgbm_pred': float(individual_preds.get('lgbm', 0)),
                'xgb_pred': float(individual_preds.get('xgb', 0)),
                'ensemble_pred': float(ensemble_pred),
                'predicted_return_pct': float(predicted_return_pct),
                'predicted_price': float(predicted_price),
                'prediction_confidence': float(confidence),
                'model_agreement': float(model_agreement),
                'signal': signal_info['signal'],
                'signal_strength': float(signal_info['strength']),
                'target_price': float(signal_info['target_price']),
                'stop_loss': float(signal_info['stop_loss'])
            }
            
            # Print results
            self._print_prediction(prediction, individual_preds)
            
            return prediction
            
        except Exception as e:
            print(f"❌ Error processing alert #{alert_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_agreement(self, predictions: List[float]) -> float:
        """Calculate model agreement score"""
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
        recent_vol = df_features['volatility_20'].iloc[-1] if 'volatility_20' in df_features else 0.02
        vol_factor = 1 / (1 + recent_vol * 10)
        
        confidence = 0.5 * agreement + 0.3 * magnitude + 0.2 * vol_factor
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_signal(self, prediction: float, confidence: float, 
                        current_price: float, df_features: pd.DataFrame) -> Dict:
        """Generate trading signal"""
        signal = 'HOLD'
        strength = 0.0
        target_price = current_price
        stop_loss = current_price
        
        # Signal thresholds
        buy_threshold = 0.002  # 0.2%
        sell_threshold = -0.002  # -0.2%
        min_confidence = 0.5
        
        if confidence >= min_confidence:
            if prediction > buy_threshold:
                signal = 'BUY'
                strength = min(prediction * confidence * 10, 1.0)
                target_price = current_price * (1 + prediction * 1.5)
                stop_loss = current_price * 0.98  # 2% stop loss
            elif prediction < sell_threshold:
                signal = 'SELL'
                strength = min(abs(prediction) * confidence * 10, 1.0)
                target_price = current_price * (1 + prediction * 1.5)
                stop_loss = current_price * 1.02  # 2% stop loss
        
        return {
            'signal': signal,
            'strength': strength,
            'target_price': target_price,
            'stop_loss': stop_loss
        }
    
    def _print_prediction(self, prediction: Dict, individual_preds: Dict):
        """Print prediction summary"""
        print(f"\n📊 PREDICTION RESULTS")
        print(f"   Current Price: ${prediction['current_price']:.2f}")
        print(f"   Ensemble Pred: {prediction['predicted_return_pct']:+.3f}%")
        print(f"   Target Price:  ${prediction['predicted_price']:.2f}")
        print(f"   Confidence:    {prediction['prediction_confidence']:.1%}")
        print(f"   Agreement:     {prediction['model_agreement']:.1%}")
        
        print(f"\n🎯 TRADING SIGNAL")
        signal_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⚪'}
        print(f"   Signal:   {signal_emoji.get(prediction['signal'], '⚪')} {prediction['signal']}")
        print(f"   Strength: {prediction['signal_strength']:.1%}")
        if prediction['signal'] != 'HOLD':
            print(f"   Target:   ${prediction['target_price']:.2f}")
            print(f"   Stop:     ${prediction['stop_loss']:.2f}")
        
        print(f"\n🤖 MODEL BREAKDOWN")
        for model_name, pred in individual_preds.items():
            print(f"   {model_name:15s}: {pred*100:+.3f}%")
    
    def run_continuous(self, interval_seconds: int = 30):
        """Run continuous monitoring and inference"""
        print(f"\n{'='*60}")
        print("🚀 Starting Watchlist Inference Engine")
        print(f"{'='*60}")
        print(f"Polling interval: {interval_seconds}s")
        print(f"Last processed ID: {self.data_reader.last_processed_id}")
        print(f"Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Fetch new alerts
                new_alerts = self.data_reader.get_new_alerts()
                
                if not new_alerts.empty:
                    print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Processing {len(new_alerts)} alerts")
                    
                    for idx, alert in new_alerts.iterrows():
                        # Process alert
                        prediction = self.process_alert(alert)
                        
                        if prediction:
                            # Write to database
                            success = self.output_writer.write_prediction(prediction)
                            if success:
                                print(f"✅ Saved prediction for Alert #{alert['alert_id']}")
                                self.data_reader.update_last_processed_id(alert['alert_id'])
                            else:
                                print(f"❌ Failed to save prediction for Alert #{alert['alert_id']}")
                        
                        # Small delay between alerts
                        time.sleep(0.5)
                
                # Wait before next check
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping inference engine...")
            print(f"Total alerts processed: {self.data_reader.last_processed_id}")
            print("Goodbye! 👋\n")
    
    def run_once(self):
        """Process all pending alerts once"""
        print(f"\n{'='*60}")
        print("🚀 Running Watchlist Inference (Single Pass)")
        print(f"{'='*60}\n")
        
        new_alerts = self.data_reader.get_new_alerts()
        
        if new_alerts.empty:
            print("📭 No new alerts to process")
            return
        
        print(f"Processing {len(new_alerts)} alerts...\n")
        
        for idx, alert in new_alerts.iterrows():
            prediction = self.process_alert(alert)
            
            if prediction:
                success = self.output_writer.write_prediction(prediction)
                if success:
                    print(f"✅ Saved prediction for Alert #{alert['alert_id']}")
                    self.data_reader.update_last_processed_id(alert['alert_id'])
        
        print(f"\n✅ Completed! Processed up to Alert #{self.data_reader.last_processed_id}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='LSTM Ensemble Inference for Watchlist Alerts'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['continuous', 'once'],
        default='continuous',
        help='Run mode: continuous monitoring or single pass'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Polling interval in seconds (continuous mode only)'
    )
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = WatchlistInferenceEngine(config_path=args.config)
    
    # Run based on mode
    if args.mode == 'continuous':
        engine.run_continuous(interval_seconds=args.interval)
    else:
        engine.run_once()


if __name__ == "__main__":
    main()
