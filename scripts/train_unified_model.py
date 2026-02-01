#!/usr/bin/env python3
"""
Train a single unified model for all symbols
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import argparse

from lstm_ensemble.complete_system import Config, LSTMEnsemble, FeatureEngineering

load_dotenv()


def read_symbols(file_path='config/symbols.txt'):
    """Read symbols from file"""
    symbols = []
    with open(file_path, 'r') as f:
        for line in f:
            symbol = line.strip()
            if symbol and not symbol.startswith('#'):
                symbols.append(symbol)
    return symbols


def load_all_symbols_data(symbols, days=10):
    """Load data for all symbols"""
    
    db_connection = os.getenv('DB_CONNECTION')
    engine = create_engine(db_connection)
    
    # Calculate trading days (roughly 6.5 hours * 60 minutes = 390 minutes per day)
    minutes_per_trading_day = 390
    total_minutes = days * minutes_per_trading_day
    
    all_data = []
    
    print("\n" + "="*70)
    print("📥 Loading data for all symbols...")
    print("="*70)
    
    for symbol in symbols:
        query = text("""
            SELECT * FROM market_data_minute
            WHERE symbol = :symbol
            ORDER BY timestamp DESC
            LIMIT :limit
        """)
        
        df = pd.read_sql(query, engine, params={'symbol': symbol, 'limit': total_minutes + 1000})
        
        if df.empty:
            print(f"⚠️  No data for {symbol} - skipping")
            continue
        
        # Reverse to chronological order
        df = df.iloc[::-1].reset_index(drop=True)
        
        print(f"✅ {symbol}: {len(df)} records ({df['timestamp'].min()} to {df['timestamp'].max()})")
        all_data.append(df)
    
    if not all_data:
        raise ValueError("No data loaded for any symbol!")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by symbol and timestamp
    combined_df = combined_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    print(f"\n📊 Combined dataset:")
    print(f"   Total records: {len(combined_df)}")
    print(f"   Symbols: {combined_df['symbol'].nunique()}")
    print(f"   Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    return combined_df


def train_unified_model(symbols, days=10, epochs=100):
    """Train a single unified model for all symbols"""
    
    print("\n" + "="*70)
    print("🤖 UNIFIED MODEL TRAINING")
    print("="*70)
    print(f"Symbols: {len(symbols)}")
    print(f"Trading days: {days}")
    print(f"Epochs: {epochs}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Load config
    config = Config()
    
    # Load data for all symbols
    df = load_all_symbols_data(symbols, days)
    
    print("\n" + "="*70)
    print("⚙️  Feature Engineering...")
    print("="*70)
    
    # Process each symbol separately for feature engineering
    processed_dfs = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        
        # Feature engineering
        symbol_df = FeatureEngineering.compute_all_features(symbol_df)
        
        # Create target
        symbol_df['target'] = symbol_df['close'].pct_change(
            periods=config._config['models']['prediction_horizon']
        ).shift(-config._config['models']['prediction_horizon'])
        
        processed_dfs.append(symbol_df)
        print(f"✅ {symbol}: {len(symbol_df)} records after features")
    
    # Combine processed data
    df = pd.concat(processed_dfs, ignore_index=True)
    df = df.dropna()
    
    print(f"\n📊 After feature engineering:")
    print(f"   Total records: {len(df)}")
    print(f"   Valid records: {df['target'].notna().sum()}")
    
    if len(df) < config._config['models']['sequence_length'] + 100:
        raise ValueError(f"Insufficient data: {len(df)} records")
    
    # Split data (80/20)
    split_idx = int(len(df) * (1 - config._config['training']['validation_split']))
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    print(f"\n📈 Data Split:")
    print(f"   Training set: {len(train_df)} records")
    print(f"   Validation set: {len(val_df)} records")
    print(f"   Training symbols: {train_df['symbol'].nunique()}")
    print(f"   Validation symbols: {val_df['symbol'].nunique()}")
    
    # Initialize ensemble
    print("\n" + "="*70)
    print("🔧 Initializing ensemble...")
    print("="*70)
    
    ensemble = LSTMEnsemble(
        feature_columns=config._config['features'],
        sequence_length=config._config['models']['sequence_length'],
        device=config._config['models']['device']
    )
    
    # Prepare sequences
    print("Preparing sequences...")
    train_data, train_targets = ensemble.prepare_sequences(train_df, 'target')
    val_data, val_targets = ensemble.prepare_sequences(val_df, 'target')
    
    print(f"   Training sequences: {len(train_data)}")
    print(f"   Validation sequences: {len(val_data)}")
    
    # Train PyTorch models
    print("\n" + "="*70)
    print("🤖 Training PyTorch models (LSTM, BiLSTM, Attention, CNN-LSTM)...")
    print("="*70)
    
    ensemble.train_pytorch_models(
        train_data, train_targets,
        val_data, val_targets,
        epochs=epochs,
        batch_size=config._config['training']['batch_size'],
        learning_rate=config._config['training']['learning_rate']
    )
    
    # Train tree models
    print("\n" + "="*70)
    print("🌲 Training tree models (LightGBM, XGBoost)...")
    print("="*70)
    
    ensemble.train_tree_models(train_data, train_targets, val_data, val_targets)
    
    # Save unified model
    model_dir = "models/UNIFIED/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    ensemble.save_models(model_dir)
    
    # Save symbol list
    import json
    with open(f"{model_dir}/symbols.json", 'w') as f:
        json.dump({
            'symbols': symbols,
            'trained_at': datetime.now().isoformat(),
            'days': days,
            'epochs': epochs,
            'total_records': len(df),
            'training_records': len(train_df),
            'validation_records': len(val_df)
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ UNIFIED MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"   Models saved to: {model_dir}")
    print(f"   Symbols trained: {len(symbols)}")
    print(f"   Total data points: {len(df)}")
    print(f"   Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Train a unified model for all symbols')
    parser.add_argument('--symbols-file', default='config/symbols.txt', help='Path to symbols file')
    parser.add_argument('--days', type=int, default=10, help='Number of trading days (default: 10)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to include (overrides file)')
    
    args = parser.parse_args()
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = read_symbols(args.symbols_file)
    
    try:
        train_unified_model(symbols, args.days, args.epochs)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
