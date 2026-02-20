#!/usr/bin/env python3
"""Train LSTM ensemble models"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lstm_ensemble.complete_system import Config, LSTMEnsemble, FeatureEngineering
from sqlalchemy import create_engine, text
import pandas as pd
import argparse
from dotenv import load_dotenv

load_dotenv()

def train(symbol: str, days: int = 30, epochs: int = 50):
    config = Config()
    
    print(f"\n{'='*60}")
    print(f"Training models for {symbol}")
    print(f"Using last {days} days of data")
    print(f"Training for {epochs} epochs")
    print(f"{'='*60}\n")
    
    # Load data
    db_connection = os.getenv('DB_CONNECTION')
    engine = create_engine(db_connection)
    
    query = text(f"""
        SELECT * FROM market_data_minute
        WHERE symbol = :symbol
        AND timestamp >= NOW() - INTERVAL '{days} days'
        ORDER BY timestamp ASC
    """)
    
    df = pd.read_sql(query, engine, params={'symbol': symbol})
    print(f"Loaded {len(df)} records")
    
    if len(df) < 500:
        print(f"❌ Insufficient data ({len(df)} records). Need at least 500.")
        return False
    
    # Feature engineering
    print("Computing features...")
    df = FeatureEngineering.compute_all_features(df)
    
    # Create target
    df['target'] = df['close'].pct_change(periods=config._config['models']['prediction_horizon']).shift(-config._config['models']['prediction_horizon'])
    df = df.dropna()
    
    if len(df) < config._config['models']['sequence_length'] + 100:
        print(f"❌ Insufficient data after feature engineering ({len(df)} records)")
        return False
    
    print(f"After feature engineering: {len(df)} records")
    
    # Split
    split_idx = int(len(df) * (1 - config._config['training']['validation_split']))
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    print(f"Training set: {len(train_df)} records")
    print(f"Validation set: {len(val_df)} records")
    
    # Initialize ensemble
    ensemble = LSTMEnsemble(
        feature_columns=config._config['features'],
        sequence_length=config._config['models']['sequence_length'],
        device=config._config['models']['device']
    )
    
    # Prepare data
    print("Preparing sequences...")
    train_data, train_targets = ensemble.prepare_sequences(train_df, 'target')
    val_data, val_targets = ensemble.prepare_sequences(val_df, 'target')
    
    # Train PyTorch models
    print("\n🤖 Training PyTorch models (LSTM, BiLSTM, Attention, CNN-LSTM)...")
    ensemble.train_pytorch_models(
        train_data, train_targets, 
        val_data, val_targets,
        epochs=epochs,
        batch_size=config._config['training']['batch_size'],
        learning_rate=config._config['training']['learning_rate']
    )
    
    # Train tree models
    print("\n🌲 Training tree models (LightGBM, XGBoost)...")
    ensemble.train_tree_models(train_data, train_targets, val_data, val_targets)
    
    # Save
    model_dir = f"models/{symbol}/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    ensemble.save_models(model_dir)
    
    print(f"\n✅ Training complete! Models saved to {model_dir}\n")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LSTM ensemble models')
    parser.add_argument('--symbol', default='AAPL', help='Stock symbol to train')
    parser.add_argument('--days', type=int, default=30, help='Days of training data (default: 30)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    args = parser.parse_args()
    
    try:
        success = train(args.symbol, args.days, args.epochs)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
