#!/usr/bin/env python3
"""
Check training and data status
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd

load_dotenv()

def check_status():
    """Check data and model status"""
    
    print("\n" + "="*60)
    print("🔍 LSTM Ensemble System Status")
    print("="*60 + "\n")
    
    db_connection = os.getenv('DB_CONNECTION')
    engine = create_engine(db_connection)
    
    # Check data in database
    print("📊 DATA STATUS")
    print("-" * 60)
    
    try:
        query = text("""
            SELECT symbol, COUNT(*) as bars, 
                   MIN(timestamp) as first_bar, 
                   MAX(timestamp) as last_bar
            FROM market_data_minute
            GROUP BY symbol
            ORDER BY symbol
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()
            
            if rows:
                for row in rows:
                    print(f"  {row.symbol:8s}: {row.bars:,} bars")
                    print(f"             from {row.first_bar} to {row.last_bar}")
            else:
                print("  ⚠️  No data in market_data_minute table")
    except Exception as e:
        print(f"  ❌ Error checking data: {e}")
    
    # Check trained models
    print("\n🤖 TRAINED MODELS")
    print("-" * 60)
    
    models_dir = Path("models")
    if models_dir.exists():
        symbols = [d.name for d in models_dir.iterdir() if d.is_dir()]
        
        if symbols:
            for symbol in sorted(symbols):
                model_path = models_dir / symbol
                files = list(model_path.glob("*"))
                
                # Check for required files
                required = [
                    'lstm_model.pth', 'bilstm_model.pth', 
                    'attention_lstm_model.pth', 'cnn_lstm_model.pth',
                    'lgbm_model.pkl', 'xgb_model.pkl', 'scaler.pkl'
                ]
                
                has_all = all((model_path / f).exists() for f in required)
                status = "✅ Complete" if has_all else "⚠️  Incomplete"
                
                print(f"  {symbol:8s}: {len(files)} files - {status}")
        else:
            print("  ⚠️  No trained models found")
    else:
        print("  ⚠️  models/ directory does not exist")
    
    # Check predictions
    print("\n📈 PREDICTIONS")
    print("-" * 60)
    
    try:
        query = text("""
            SELECT symbol, 
                   current_price,
                   ensemble_pred,
                   predicted_return_pct,
                   signal,
                   prediction_confidence,
                   updated_at
            FROM public.lstm_ensemble_output
            ORDER BY updated_at DESC
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()
            
            if rows:
                for row in rows:
                    signal_emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⚪'}
                    print(f"  {row.symbol:8s}: ${row.current_price:.2f} → {row.predicted_return_pct:+.2f}%")
                    print(f"             {signal_emoji.get(row.signal, '⚪')} {row.signal} (confidence: {row.prediction_confidence:.1%})")
                    print(f"             Updated: {row.updated_at}")
                    print()
            else:
                print("  ⚠️  No predictions yet")
    except Exception as e:
        print(f"  ⚠️  lstm_ensemble_output table not ready: {e}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    check_status()
