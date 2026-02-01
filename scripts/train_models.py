#!/usr/bin/env python3
"""Train LSTM ensemble models"""

import sys
sys.path.insert(0, '..')

from lstm_ensemble import Config, LSTMEnsemble, FeatureEngineering
from sqlalchemy import create_engine, text
import pandas as pd
import argparse

def train(symbol: str, days: int = 30):
    config = Config()
    
    print(f"Training models for {symbol} using last {days} days...")
    
    # Load data
    engine = create_engine(config.db_connection)
    query = f"""
    SELECT * FROM {config.market_data_table}
    WHERE symbol = :symbol
    AND timestamp >= NOW() - INTERVAL '{days} days'
    ORDER BY timestamp ASC
    """
    
    df = pd.read_sql(text(query), engine, params={'symbol': symbol})
    print(f"Loaded {len(df)} records")
    
    # Feature engineering
    df = FeatureEngineering.compute_all_features(df)
    
    # Create target
    df['target'] = df['close'].pct_change(periods=5).shift(-5)
    df = df.dropna()
    
    # Split
    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    # Initialize ensemble
    ensemble = LSTMEnsemble(
        feature_columns=config.features,
        sequence_length=config.sequence_length,
        device=config.device
    )
    
    # Prepare data
    train_data, train_targets = ensemble.prepare_sequences(train_df, 'target')
    val_data, val_targets = ensemble.prepare_sequences(val_df, 'target')
    
    # Train
    ensemble.train_pytorch_models(train_data, train_targets, val_data, val_targets, epochs=50)
    ensemble.train_tree_models(train_data, train_targets, val_data, val_targets)
    
    # Save
    ensemble.save_models(f"models/{symbol}/")
    
    print(f"✓ Training complete! Models saved to models/{symbol}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='AAPL', help='Stock symbol')
    parser.add_argument('--days', type=int, default=30, help='Days of training data')
    args = parser.parse_args()
    
    train(args.symbol, args.days)
