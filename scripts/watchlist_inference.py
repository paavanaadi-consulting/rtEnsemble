#!/usr/bin/env python3
"""
Watchlist Alerts LSTM Ensemble Inference
=========================================

Reads latest records from public.watchlist_alerts, builds sequences of
the last 3 alerts per symbol, runs through the watchlist-trained LSTM
ensemble, and writes results to public.lstm_ensemble_output.

Modes:
    continuous - Poll for new alerts every N seconds
    once       - Process pending alerts once and exit
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import yaml
from dotenv import load_dotenv

from sqlalchemy import create_engine, text

load_dotenv()

from lstm_ensemble.watchlist_features import (
    extract_alert_features,
    WATCHLIST_FEATURE_COLUMNS,
)
from lstm_ensemble.watchlist_ensemble import WatchlistEnsemble


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

class WatchlistDataReader:
    """Reads data from watchlist_alerts table."""

    def __init__(self, db_connection_string: str):
        self.engine = create_engine(db_connection_string)
        self.last_processed_id = self._get_last_processed_id()

    def _get_last_processed_id(self) -> int:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT COALESCE(MAX(watchlist_alert_id), 0) "
                    "FROM public.lstm_ensemble_output"
                ))
                return result.scalar() or 0
        except Exception as e:
            print(f"Warning: Could not get last processed ID: {e}")
            return 0

    def get_new_alerts(self) -> pd.DataFrame:
        """Fetch new alerts that haven't been processed yet."""
        try:
            query = text("""
                SELECT id as alert_id, symbol, alert_type, price, volume,
                       details, timestamp
                FROM public.watchlist_alerts
                WHERE id > :last_id
                ORDER BY id ASC
                LIMIT 100
            """)
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={"last_id": self.last_processed_id})
            if not df.empty:
                print(f"Found {len(df)} new alerts "
                      f"(IDs: {df['alert_id'].min()}-{df['alert_id'].max()})")
            return df
        except Exception as e:
            print(f"Error fetching alerts: {e}")
            return pd.DataFrame()

    def get_last_n_alerts(self, symbol: str, n: int = 3,
                          before_id: int = None) -> pd.DataFrame:
        """Get the last N alerts for a symbol (for building the sequence)."""
        try:
            if before_id:
                query = text("""
                    SELECT id, symbol, alert_type, price, volume, details, timestamp
                    FROM public.watchlist_alerts
                    WHERE symbol = :symbol AND id <= :before_id
                        AND price IS NOT NULL AND price::float > 0
                    ORDER BY id DESC
                    LIMIT :n
                """)
                params = {"symbol": symbol, "before_id": before_id, "n": n}
            else:
                query = text("""
                    SELECT id, symbol, alert_type, price, volume, details, timestamp
                    FROM public.watchlist_alerts
                    WHERE symbol = :symbol
                        AND price IS NOT NULL AND price::float > 0
                    ORDER BY id DESC
                    LIMIT :n
                """)
                params = {"symbol": symbol, "n": n}

            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params=params)

            # Reverse to chronological order
            if not df.empty:
                df = df.iloc[::-1].reset_index(drop=True)
            return df
        except Exception as e:
            print(f"Error fetching alerts for {symbol}: {e}")
            return pd.DataFrame()

    def update_last_processed_id(self, alert_id: int):
        if alert_id > self.last_processed_id:
            self.last_processed_id = alert_id


class EnsembleOutputWriter:
    """Writes ensemble predictions to lstm_ensemble_output table."""

    def __init__(self, db_connection_string: str):
        self.engine = create_engine(db_connection_string)
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        create_sql = text("""
            CREATE TABLE IF NOT EXISTS public.lstm_ensemble_output (
                symbol VARCHAR(10) PRIMARY KEY,
                watchlist_alert_id BIGINT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                alert_type VARCHAR(50),
                current_price DECIMAL(12, 4),
                lstm_pred DECIMAL(10, 6),
                bilstm_pred DECIMAL(10, 6),
                attention_lstm_pred DECIMAL(10, 6),
                cnn_lstm_pred DECIMAL(10, 6),
                lgbm_pred DECIMAL(10, 6),
                xgb_pred DECIMAL(10, 6),
                ensemble_pred DECIMAL(10, 6),
                predicted_return_pct DECIMAL(10, 4),
                predicted_price DECIMAL(12, 4),
                prediction_confidence DECIMAL(5, 4),
                model_agreement DECIMAL(5, 4),
                signal VARCHAR(10),
                signal_strength DECIMAL(5, 4),
                target_price DECIMAL(12, 4),
                stop_loss DECIMAL(12, 4),
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
                conn.execute(create_sql)
                conn.commit()
        except Exception as e:
            print(f"Table creation warning: {e}")

    def write_prediction(self, prediction: Dict) -> bool:
        try:
            sql = text("""
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
                conn.execute(sql, prediction)
                conn.commit()
            return True
        except Exception as e:
            print(f"Error writing prediction: {e}")
            return False


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class WatchlistInferenceEngine:
    """Inference engine using watchlist-trained LSTM ensemble."""

    def __init__(self, model_dir: str = "models/watchlist", seq_len: int = 3):
        self.seq_len = seq_len
        self.model_dir = model_dir

        db_connection = os.getenv("DB_CONNECTION")
        if not db_connection:
            raise ValueError("DB_CONNECTION environment variable not set!")

        self.data_reader = WatchlistDataReader(db_connection)
        self.output_writer = EnsembleOutputWriter(db_connection)

        # Load the unified watchlist model
        self.ensemble = None
        self._load_model()

        print("Watchlist Inference Engine initialized")

    def _load_model(self):
        """Load the watchlist-trained ensemble."""
        model_path = Path(self.model_dir)
        if not model_path.exists():
            print(f"WARNING: No watchlist model at {model_path}. "
                  f"Run train_from_watchlist.py first.")
            return

        device = os.getenv("DEVICE", "cpu")
        self.ensemble = WatchlistEnsemble(
            seq_len=self.seq_len,
            device=device,
        )
        self.ensemble.load_models(str(model_path))

    def process_alert(self, alert: pd.Series) -> Optional[Dict]:
        """Process a single new watchlist alert."""
        symbol = alert["symbol"]
        alert_id = alert["alert_id"]
        timestamp = pd.to_datetime(alert["timestamp"])

        print(f"\n{'='*60}")
        print(f"Processing Alert #{alert_id}: {symbol} @ {timestamp}")

        if self.ensemble is None:
            print("No watchlist model loaded. Skipping.")
            return None

        # Get the last N alerts for this symbol (including current one)
        recent_alerts = self.data_reader.get_last_n_alerts(
            symbol, n=self.seq_len, before_id=alert_id
        )

        if len(recent_alerts) < self.seq_len:
            print(f"Only {len(recent_alerts)} alerts for {symbol}, "
                  f"need {self.seq_len}. Skipping.")
            return None

        # Extract features from each alert
        feature_rows = []
        prev_price = None
        for _, row in recent_alerts.iterrows():
            row_dict = row.to_dict()
            feats = extract_alert_features(row_dict)
            # Compute price_change_pct
            curr_price = float(row_dict.get("price", 0) or 0)
            if prev_price and prev_price > 0 and curr_price > 0:
                feats["price_change_pct"] = (curr_price - prev_price) / prev_price
            prev_price = curr_price
            feature_rows.append(feats)

        # Build feature array
        seq = np.array(
            [[row[col] for col in WATCHLIST_FEATURE_COLUMNS] for row in feature_rows],
            dtype=np.float32,
        )
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        # Get current price
        current_price = float(alert.get("price", 0) or 0)
        if current_price <= 0:
            current_price = float(recent_alerts["price"].iloc[-1] or 0)
        alert_type = alert.get("alert_type", "")

        print(f"Current Price: ${current_price:.2f}")
        print(f"Alert Type: {alert_type}")

        # Predict
        result = self.ensemble.predict_single(seq)
        ens_pred = result["ensemble_pred"]
        individual = result["individual_preds"]
        confidence = result["confidence"]
        agreement = result["agreement"]
        signal = result["signal"]
        strength = result["signal_strength"]

        predicted_return_pct = ens_pred * 100
        predicted_price = current_price * (1 + ens_pred) if current_price > 0 else 0

        # Target / stop loss
        if signal == "BUY":
            target_price = current_price * (1 + abs(ens_pred) * 2)
            stop_loss = current_price * 0.97  # 3% stop for small caps
        elif signal == "SELL":
            target_price = current_price * (1 - abs(ens_pred) * 2)
            stop_loss = current_price * 1.03
        else:
            target_price = current_price
            stop_loss = current_price

        prediction = {
            "symbol": symbol,
            "watchlist_alert_id": int(alert_id),
            "timestamp": timestamp,
            "alert_type": alert_type,
            "current_price": current_price,
            "lstm_pred": float(individual.get("lstm", 0)),
            "bilstm_pred": float(individual.get("bilstm", 0)),
            "attention_lstm_pred": float(individual.get("attention", 0)),
            "cnn_lstm_pred": float(individual.get("mlp", 0)),  # mlp maps to cnn_lstm column
            "lgbm_pred": float(individual.get("lgbm", 0)),
            "xgb_pred": float(individual.get("xgb", 0)),
            "ensemble_pred": float(ens_pred),
            "predicted_return_pct": float(predicted_return_pct),
            "predicted_price": float(predicted_price),
            "prediction_confidence": float(confidence),
            "model_agreement": float(agreement),
            "signal": signal,
            "signal_strength": float(strength),
            "target_price": float(target_price),
            "stop_loss": float(stop_loss),
        }

        self._print_prediction(prediction, individual)
        return prediction

    def _print_prediction(self, pred: Dict, individual: Dict):
        print(f"\nPREDICTION RESULTS")
        print(f"  Current Price: ${pred['current_price']:.2f}")
        print(f"  Ensemble Pred: {pred['predicted_return_pct']:+.3f}%")
        print(f"  Target Price:  ${pred['predicted_price']:.2f}")
        print(f"  Confidence:    {pred['prediction_confidence']:.1%}")
        print(f"  Agreement:     {pred['model_agreement']:.1%}")

        signal_marker = {"BUY": "[BUY]", "SELL": "[SELL]", "HOLD": "[HOLD]"}
        print(f"\nTRADING SIGNAL")
        print(f"  Signal:   {signal_marker.get(pred['signal'], '[?]')} {pred['signal']}")
        print(f"  Strength: {pred['signal_strength']:.1%}")
        if pred["signal"] != "HOLD":
            print(f"  Target:   ${pred['target_price']:.2f}")
            print(f"  Stop:     ${pred['stop_loss']:.2f}")

        print(f"\nMODEL BREAKDOWN")
        for name, val in individual.items():
            print(f"  {name:15s}: {val*100:+.3f}%")

    def run_continuous(self, interval_seconds: int = 30):
        """Run continuous monitoring."""
        print(f"\n{'='*60}")
        print("Starting Watchlist Inference Engine")
        print(f"{'='*60}")
        print(f"Polling interval: {interval_seconds}s")
        print(f"Model: {self.model_dir}")
        print(f"Last processed ID: {self.data_reader.last_processed_id}")
        print(f"Press Ctrl+C to stop\n")

        try:
            while True:
                new_alerts = self.data_reader.get_new_alerts()

                if not new_alerts.empty:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\n[{ts}] Processing {len(new_alerts)} alerts")

                    for _, alert in new_alerts.iterrows():
                        prediction = self.process_alert(alert)
                        if prediction:
                            success = self.output_writer.write_prediction(prediction)
                            if success:
                                print(f"Saved prediction for Alert #{alert['alert_id']}")
                                self.data_reader.update_last_processed_id(alert["alert_id"])
                        time.sleep(0.1)

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print(f"\n\nStopping inference engine.")
            print(f"Last processed: {self.data_reader.last_processed_id}")

    def run_once(self):
        """Process all pending alerts once."""
        print(f"\n{'='*60}")
        print("Running Watchlist Inference (Single Pass)")
        print(f"{'='*60}\n")

        new_alerts = self.data_reader.get_new_alerts()
        if new_alerts.empty:
            print("No new alerts to process.")
            return

        print(f"Processing {len(new_alerts)} alerts...\n")
        for _, alert in new_alerts.iterrows():
            prediction = self.process_alert(alert)
            if prediction:
                success = self.output_writer.write_prediction(prediction)
                if success:
                    print(f"Saved prediction for Alert #{alert['alert_id']}")
                    self.data_reader.update_last_processed_id(alert["alert_id"])

        print(f"\nCompleted! Processed up to Alert #{self.data_reader.last_processed_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LSTM Ensemble Inference for Watchlist Alerts"
    )
    parser.add_argument(
        "--model-dir", type=str, default="models/watchlist",
        help="Path to watchlist model directory (default: models/watchlist)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=3,
        help="Number of past alerts per sequence (default: 3)"
    )
    parser.add_argument(
        "--mode", type=str, choices=["continuous", "once"], default="continuous",
        help="Run mode (default: continuous)"
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Polling interval in seconds (default: 30)"
    )

    args = parser.parse_args()

    engine = WatchlistInferenceEngine(
        model_dir=args.model_dir,
        seq_len=args.seq_len,
    )

    if args.mode == "continuous":
        engine.run_continuous(interval_seconds=args.interval)
    else:
        engine.run_once()


if __name__ == "__main__":
    main()
