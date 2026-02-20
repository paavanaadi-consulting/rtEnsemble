#!/usr/bin/env python3
"""
Run inference for a specific symbol
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
        print(f"❌ No models found for {symbol}. Please train first.")
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
    
    if len(df) < config._config['models']['sequence_length'] + 100:
        print(f"❌ Insufficient data ({len(df)} records). Need at least {config._config['models']['sequence_length'] + 100}.")
        return False
    
    # Reverse to chronological order
    df = df.iloc[::-1].reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} records")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Feature engineering
    print("\nComputing features...")
    df = FeatureEngineering.compute_all_features(df)
    df = df.dropna()
    
    print(f"✅ Features computed ({len(df)} records after cleanup)")
    
    # Load ensemble
    print("\nLoading ensemble models...")
    ensemble = LSTMEnsemble(
        feature_columns=config._config['features'],
        sequence_length=config._config['models']['sequence_length'],
        device=config._config['models']['device']
    )
    
    ensemble.load_models(model_dir)
    print("✅ Models loaded")
    
    # Prepare the most recent sequence
    print("\nPreparing prediction sequence...")
    feature_data = df[config._config['features']].values[-config._config['models']['sequence_length']:]
    
    if len(feature_data) < config._config['models']['sequence_length']:
        print(f"❌ Not enough data for sequence ({len(feature_data)} < {config._config['models']['sequence_length']})")
        return False
    
    # Make prediction
    print("🔮 Making prediction...")
    individual_predictions = ensemble.predict(feature_data)
    ensemble_prediction = ensemble.predict_ensemble(feature_data)
    
    # Get latest price
    latest_price = float(df['close'].iloc[-1])
    latest_timestamp = df['timestamp'].iloc[-1]
    
    # Calculate confidence metrics
    model_preds = list(individual_predictions.values())
    pred_std = np.std(model_preds) if len(model_preds) > 1 else 0
    pred_mean = np.mean(model_preds) if model_preds else 0
    agreement = max(0, 1 - (pred_std / (abs(pred_mean) + 1e-10)))
    magnitude = min(abs(ensemble_prediction) * 100, 1.0)
    confidence = min(0.5 * agreement + 0.5 * magnitude, 1.0)
    signal_strength_label = 'STRONG' if confidence > 0.7 else 'MODERATE' if confidence > 0.3 else 'WEAK'
    
    print(f"\n{'='*60}")
    print(f"📊 PREDICTION RESULTS for {symbol}")
    print(f"{'='*60}")
    print(f"Latest Price: ${latest_price:.2f}")
    print(f"Latest Time:  {latest_timestamp}")
    print(f"Ensemble Pred: {ensemble_prediction:.6f}")
    print(f"Direction:    {'📈 UP' if ensemble_prediction > 0 else '📉 DOWN'}")
    print(f"Confidence:   {confidence:.2%}")
    print(f"Strength:     {signal_strength_label}")
    print(f"\nIndividual Model Predictions:")
    for model_name, pred in individual_predictions.items():
        print(f"  {model_name:15s}: {pred:.6f}")
    print(f"{'='*60}\n")
    
    # Write to database
    print("Writing prediction to database...")
    
    # Calculate additional metrics
    predicted_price = latest_price * (1 + ensemble_prediction)
    signal = 'BUY' if ensemble_prediction > 0.002 else 'SELL' if ensemble_prediction < -0.002 else 'HOLD'
    model_agreement = agreement
    
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
    
    parser = argparse.ArgumentParser(description='Run inference for a symbol')
    parser.add_argument('symbol', help='Stock symbol to predict')
    
    args = parser.parse_args()
    
    try:
        success = run_inference_for_symbol(args.symbol)
        if success:
            print("\n✅ Inference completed successfully!\n")
        else:
            print("\n❌ Inference failed.\n")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
