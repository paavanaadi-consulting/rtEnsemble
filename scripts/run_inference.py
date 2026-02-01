#!/usr/bin/env python3
"""Run live inference"""

import sys
sys.path.insert(0, '..')

from lstm_ensemble import Config, LSTMEnsemble, LiveDataReader, LiveInferenceEngine
import argparse

def run_inference(symbol: str):
    config = Config()
    
    print(f"Starting live inference for {symbol}...")
    
    # Load trained ensemble
    ensemble = LSTMEnsemble(
        feature_columns=config.features,
        sequence_length=config.sequence_length,
        device=config.device
    )
    ensemble.load_models(f"models/{symbol}/")
    
    # Data reader
    data_reader = LiveDataReader(
        db_connection_string=config.db_connection,
        table_name=config.market_data_table,
        symbol=symbol,
        lookback_minutes=500
    )
    
    # Inference engine
    engine = LiveInferenceEngine(
        ensemble=ensemble,
        data_reader=data_reader,
        feature_columns=config.features,
        db_connection_string=config.db_connection,
        enable_signals=True
    )
    
    # Start
    engine.start_live_inference(interval_seconds=60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='AAPL', help='Stock symbol')
    args = parser.parse_args()
    
    run_inference(args.symbol)
