#!/usr/bin/env python3
"""
Train models for multiple symbols from symbols.txt
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime

# Read symbols from file
def read_symbols(file_path='config/symbols.txt'):
    """Read symbols from file"""
    symbols = []
    with open(file_path, 'r') as f:
        for line in f:
            symbol = line.strip()
            if symbol and not symbol.startswith('#'):
                symbols.append(symbol)
    return symbols


def train_symbol(symbol: str, days: int, epochs: int):
    """Train a single symbol"""
    print(f"\n{'='*70}")
    print(f"🚀 Starting training for {symbol}")
    print(f"{'='*70}\n")
    
    # Import here to avoid issues
    from lstm_ensemble.complete_system import Config, LSTMEnsemble, FeatureEngineering
    from sqlalchemy import create_engine, text
    import pandas as pd
    from dotenv import load_dotenv
    
    load_dotenv()
    
    config = Config()
    
    # Load data
    db_connection = os.getenv('DB_CONNECTION')
    engine = create_engine(db_connection)
    
    # Calculate trading days (roughly 6.5 hours * 60 minutes = 390 minutes per day)
    minutes_per_trading_day = 390
    total_minutes = days * minutes_per_trading_day
    
    query = text(f"""
        SELECT * FROM market_data_minute
        WHERE symbol = :symbol
        ORDER BY timestamp DESC
        LIMIT :limit
    """)
    
    df = pd.read_sql(query, engine, params={'symbol': symbol, 'limit': total_minutes + 1000})
    
    if df.empty:
        print(f"❌ No data found for {symbol}")
        return False
    
    # Reverse to chronological order
    df = df.iloc[::-1].reset_index(drop=True)
    
    print(f"Loaded {len(df)} records for {symbol}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
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
    print("\n🤖 Training PyTorch models...")
    ensemble.train_pytorch_models(
        train_data, train_targets, 
        val_data, val_targets,
        epochs=epochs,
        batch_size=config._config['training']['batch_size'],
        learning_rate=config._config['training']['learning_rate']
    )
    
    # Train tree models
    print("\n🌲 Training tree models...")
    ensemble.train_tree_models(train_data, train_targets, val_data, val_targets)
    
    # Save
    model_dir = f"models/{symbol}/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    ensemble.save_models(model_dir)
    
    print(f"\n✅ Training complete for {symbol}! Models saved to {model_dir}\n")
    return True


def main():
    parser = argparse.ArgumentParser(description='Train models for multiple symbols')
    parser.add_argument('--symbols-file', default='config/symbols.txt', help='Path to symbols file')
    parser.add_argument('--days', type=int, default=10, help='Number of trading days (default: 10)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to train (overrides file)')
    
    args = parser.parse_args()
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = read_symbols(args.symbols_file)
    
    print("\n" + "="*70)
    print("📊 MULTI-SYMBOL TRAINING")
    print("="*70)
    print(f"Symbols to train: {len(symbols)}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Trading days: {args.days}")
    print(f"Epochs: {args.epochs}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    success_count = 0
    failed_symbols = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
        try:
            if train_symbol(symbol, args.days, args.epochs):
                success_count += 1
            else:
                failed_symbols.append(symbol)
        except Exception as e:
            print(f"❌ Error training {symbol}: {e}")
            import traceback
            traceback.print_exc()
            failed_symbols.append(symbol)
    
    # Final summary
    print("\n" + "="*70)
    print("📊 TRAINING SUMMARY")
    print("="*70)
    print(f"Total symbols: {len(symbols)}")
    print(f"Successfully trained: {success_count}")
    print(f"Failed: {len(failed_symbols)}")
    if failed_symbols:
        print(f"Failed symbols: {', '.join(failed_symbols)}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
