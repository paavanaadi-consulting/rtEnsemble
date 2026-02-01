#!/usr/bin/env python3
"""
Run inference for all symbols that have data in the database
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
import torch
import json

from lstm_ensemble.complete_system import Config, LSTMEnsemble, FeatureEngineering

load_dotenv()


def get_symbols_from_database():
    """Get list of symbols that have data in the database"""
    db_connection = os.getenv('DB_CONNECTION')
    engine = create_engine(db_connection)
    
    query = text("""
        SELECT DISTINCT symbol 
        FROM market_data_minute 
        ORDER BY symbol
    """)
    
    df = pd.read_sql(query, engine)
    symbols = df['symbol'].tolist()
    
    return symbols


def run_inference_for_symbol(symbol: str):
    """Run inference for a specific symbol"""
    
    print(f"\n{'='*60}")
    print(f"🔮 Running inference for {symbol}")
    print(f"{'='*60}\n")
    
    # Load config
    config = Config()
    
    # Database connection
    db_connection = os.getenv('DB_CONNECTION')
    engine = create_engine(db_connection)
    
    # Check if models exist
    model_dir = f"models/{symbol}/"
    if not Path(model_dir).exists():
        print(f"⚠️  No models found for {symbol}. Skipping...")
        return False
    
    print(f"✅ Found models in {model_dir}")
    
    # Load historical data
    print(f"Loading historical data for {symbol}...")
    query = text("""
        SELECT * FROM market_data_minute
        WHERE symbol = :symbol
        ORDER BY timestamp DESC
        LIMIT 5000
    """)
    
    df = pd.read_sql(query, engine, params={'symbol': symbol})
    
    if df.empty:
        print(f"⚠️  No data found for {symbol}. Skipping...")
        return False
    
    if len(df) < config._config['models']['sequence_length'] + 100:
        print(f"⚠️  Insufficient data ({len(df)} records). Skipping...")
        return False
    
    # Reverse to chronological order
    df = df.iloc[::-1].reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} records")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Feature engineering
    print("Computing features...")
    df = FeatureEngineering.compute_all_features(df)
    df = df.dropna()
    
    print(f"✅ Features computed ({len(df)} records after cleanup)")
    
    # Load ensemble
    print("Loading ensemble models...")
    ensemble = LSTMEnsemble(
        feature_columns=config._config['features'],
        sequence_length=config._config['models']['sequence_length'],
        device=config._config['models']['device']
    )
    
    ensemble.load_models(model_dir)
    print("✅ Models loaded")
    
    # Prepare the most recent sequence
    print("Preparing prediction sequence...")
    feature_data = df[config._config['features']].values[-config._config['models']['sequence_length']:]
    
    if len(feature_data) < config._config['models']['sequence_length']:
        print(f"⚠️  Not enough data for sequence. Skipping...")
        return False
    
    # Make prediction
    print("🔮 Making prediction...")
    individual_predictions = ensemble.predict(feature_data)
    ensemble_prediction = ensemble.predict_ensemble(feature_data)
    
    # Get latest price
    latest_price = float(df['close'].iloc[-1])
    latest_timestamp = df['timestamp'].iloc[-1]
    
    # Calculate confidence metrics
    confidence = abs(ensemble_prediction) * 100
    signal_strength = 'STRONG' if confidence > 0.7 else 'MODERATE' if confidence > 0.3 else 'WEAK'
    
    print(f"\n{'='*60}")
    print(f"📊 PREDICTION RESULTS for {symbol}")
    print(f"{'='*60}")
    print(f"Latest Price: ${latest_price:.2f}")
    print(f"Latest Time:  {latest_timestamp}")
    print(f"Ensemble Pred: {ensemble_prediction:.6f}")
    print(f"Direction:    {'📈 UP' if ensemble_prediction > 0 else '📉 DOWN'}")
    print(f"Confidence:   {confidence:.2f}%")
    print(f"Strength:     {signal_strength}")
    print(f"{'='*60}\n")
    
    # Write to database
    print("Writing prediction to database...")
    
    # Calculate additional metrics
    predicted_price = latest_price * (1 + ensemble_prediction)
    signal = 'BUY' if ensemble_prediction > 0.005 else 'SELL' if ensemble_prediction < -0.005 else 'HOLD'
    model_agreement = np.std([v for v in individual_predictions.values()])  # Lower = better agreement
    
    upsert_query = text("""
        INSERT INTO lstm_ensemble_output (
            symbol, timestamp, current_price,
            lstm_pred, bilstm_pred, attention_lstm_pred, cnn_lstm_pred,
            lgbm_pred, xgb_pred,
            ensemble_pred, predicted_return_pct, predicted_price,
            prediction_confidence, model_agreement,
            signal, signal_strength,
            created_at, updated_at
        )
        VALUES (
            :symbol, :timestamp, :current_price,
            :lstm_pred, :bilstm_pred, :attention_lstm_pred, :cnn_lstm_pred,
            :lgbm_pred, :xgb_pred,
            :ensemble_pred, :predicted_return_pct, :predicted_price,
            :prediction_confidence, :model_agreement,
            :signal, :signal_strength,
            NOW(), NOW()
        )
        ON CONFLICT (symbol) 
        DO UPDATE SET
            timestamp = EXCLUDED.timestamp,
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
            updated_at = EXCLUDED.updated_at
    """)
    
    with engine.connect() as conn:
        conn.execute(upsert_query, {
            'symbol': symbol,
            'timestamp': latest_timestamp,
            'current_price': latest_price,
            'lstm_pred': float(individual_predictions.get('lstm', 0)),
            'bilstm_pred': float(individual_predictions.get('bilstm', 0)),
            'attention_lstm_pred': float(individual_predictions.get('attention_lstm', 0)),
            'cnn_lstm_pred': float(individual_predictions.get('cnn_lstm', 0)),
            'lgbm_pred': float(individual_predictions.get('lgbm', 0)),
            'xgb_pred': float(individual_predictions.get('xgb', 0)),
            'ensemble_pred': float(ensemble_prediction),
            'predicted_return_pct': float(ensemble_prediction * 100),
            'predicted_price': float(predicted_price),
            'prediction_confidence': float(confidence),
            'model_agreement': float(model_agreement),
            'signal': signal,
            'signal_strength': float(confidence)
        })
        conn.commit()
    
    print("✅ Prediction saved to lstm_ensemble_output")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference for all symbols in database')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to run inference on (optional)')
    
    args = parser.parse_args()
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
        print(f"\n📋 Using provided symbols: {', '.join(symbols)}\n")
    else:
        print("\n🔍 Fetching symbols from database...\n")
        symbols = get_symbols_from_database()
        print(f"✅ Found {len(symbols)} symbols in database")
        print(f"   Symbols: {', '.join(symbols)}\n")
    
    print("="*60)
    print("🔮 BATCH INFERENCE")
    print("="*60)
    print(f"Total symbols: {len(symbols)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    success_count = 0
    skipped_count = 0
    failed_symbols = []
    successful_symbols = []
    skipped_symbols = []
    
    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
        try:
            result = run_inference_for_symbol(symbol)
            if result:
                success_count += 1
                successful_symbols.append(symbol)
            else:
                skipped_count += 1
                skipped_symbols.append(symbol)
        except Exception as e:
            print(f"❌ Error running inference for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            failed_symbols.append(symbol)
    
    # Final summary
    print("\n" + "="*60)
    print("📊 INFERENCE SUMMARY")
    print("="*60)
    print(f"Total symbols: {len(symbols)}")
    print(f"Successfully predicted: {success_count}")
    print(f"Skipped (no models/data): {skipped_count}")
    print(f"Failed with errors: {len(failed_symbols)}")
    
    if successful_symbols:
        print(f"\n✅ Successful symbols ({len(successful_symbols)}):")
        print(f"   {', '.join(successful_symbols)}")
    
    if skipped_symbols:
        print(f"\n⚠️  Skipped symbols ({len(skipped_symbols)}):")
        print(f"   {', '.join(skipped_symbols)}")
    
    if failed_symbols:
        print(f"\n❌ Failed symbols ({len(failed_symbols)}):")
        print(f"   {', '.join(failed_symbols)}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
