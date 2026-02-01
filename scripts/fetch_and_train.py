#!/usr/bin/env python3
"""
Fetch historical data from Polygon API and train LSTM ensemble models
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List
import time
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import requests

# Load environment
load_dotenv()

# Import after env is loaded
from lstm_ensemble.complete_system import Config, LSTMEnsemble, FeatureEngineering


class PolygonDataFetcher:
    """Fetch historical data from Polygon API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
    
    def fetch_aggregates(self, symbol: str, from_date: str, to_date: str, 
                        timespan: str = 'minute', multiplier: int = 1) -> pd.DataFrame:
        """
        Fetch aggregate bars from Polygon
        
        Args:
            symbol: Stock symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            timespan: minute, hour, day, week, month
            multiplier: Size of timespan multiplier
        """
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000,
            'apiKey': self.api_key
        }
        
        print(f"Fetching {symbol} data from {from_date} to {to_date}...")
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'results' not in data or not data['results']:
                print(f"⚠️  No data returned for {symbol}")
                return pd.DataFrame()
            
            results = data['results']
            
            df = pd.DataFrame(results)
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df['symbol'] = symbol
            
            # Rename columns to match our schema
            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vw': 'vwap',
                'n': 'transactions'
            })
            
            # Select relevant columns
            df = df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']]
            
            print(f"✅ Fetched {len(df)} bars for {symbol}")
            return df
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"⚠️  Rate limit hit. Waiting 60 seconds...")
                time.sleep(60)
                return self.fetch_aggregates(symbol, from_date, to_date, timespan, multiplier)
            else:
                print(f"❌ HTTP Error fetching {symbol}: {e}")
                return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_symbols(self, symbols: List[str], days_back: int = 60) -> pd.DataFrame:
        """Fetch data for multiple symbols"""
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        from_str = from_date.strftime('%Y-%m-%d')
        to_str = to_date.strftime('%Y-%m-%d')
        
        all_data = []
        
        for symbol in symbols:
            df = self.fetch_aggregates(symbol, from_str, to_str, 'minute', 1)
            if not df.empty:
                all_data.append(df)
            
            # Rate limit: 5 requests per minute for free tier
            time.sleep(12)  # Wait 12 seconds between requests
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            return combined
        else:
            return pd.DataFrame()


def save_to_database(df: pd.DataFrame, db_connection: str, table_name: str = 'market_data_minute'):
    """Save data to database"""
    if df.empty:
        print("No data to save")
        return
    
    engine = create_engine(db_connection)
    
    # Create table if not exists
    create_table_sql = text("""
        CREATE TABLE IF NOT EXISTS market_data_minute (
            id BIGSERIAL PRIMARY KEY,
            symbol VARCHAR(10) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open DECIMAL(12, 4),
            high DECIMAL(12, 4),
            low DECIMAL(12, 4),
            close DECIMAL(12, 4),
            volume BIGINT,
            vwap DECIMAL(12, 4),
            transactions INTEGER,
            received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        );
        
        CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
            ON market_data_minute(symbol, timestamp DESC);
    """)
    
    with engine.connect() as conn:
        conn.execute(create_table_sql)
        conn.commit()
    
    # Insert data
    print(f"Saving {len(df)} records to database...")
    df.to_sql(
        table_name,
        engine,
        if_exists='append',
        index=False,
        method='multi',
        chunksize=1000
    )
    print("✅ Data saved to database")


def train_model_for_symbol(symbol: str, config: Config, days: int = 30):
    """Train LSTM ensemble for a symbol"""
    print(f"\n{'='*60}")
    print(f"Training models for {symbol}")
    print(f"{'='*60}\n")
    
    # Load data from database
    engine = create_engine(config._config['database']['connection_string'])
    
    query = text(f"""
        SELECT * FROM market_data_minute
        WHERE symbol = :symbol
        AND timestamp >= NOW() - INTERVAL '{days} days'
        ORDER BY timestamp ASC
    """)
    
    df = pd.read_sql(query, engine, params={'symbol': symbol})
    
    if len(df) < 500:
        print(f"⚠️  Insufficient data for {symbol} ({len(df)} records). Need at least 500.")
        return False
    
    print(f"Loaded {len(df)} records")
    
    # Feature engineering
    print("Computing features...")
    df = FeatureEngineering.compute_all_features(df)
    
    # Create target (5-bar future return)
    df['target'] = df['close'].pct_change(periods=config._config['models']['prediction_horizon']).shift(-config._config['models']['prediction_horizon'])
    df = df.dropna()
    
    if len(df) < config._config['models']['sequence_length'] + 100:
        print(f"⚠️  Insufficient data after feature engineering ({len(df)} records)")
        return False
    
    print(f"After feature engineering: {len(df)} records")
    
    # Split data
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
    
    # Prepare sequences
    print("Preparing sequences...")
    train_data, train_targets = ensemble.prepare_sequences(train_df, 'target')
    val_data, val_targets = ensemble.prepare_sequences(val_df, 'target')
    
    # Train PyTorch models
    print("\n🤖 Training PyTorch models (LSTM, BiLSTM, Attention, CNN-LSTM)...")
    ensemble.train_pytorch_models(
        train_data, train_targets, 
        val_data, val_targets,
        epochs=config._config['training']['epochs'],
        batch_size=config._config['training']['batch_size'],
        learning_rate=config._config['training']['learning_rate']
    )
    
    # Train tree models
    print("\n🌲 Training tree models (LightGBM, XGBoost)...")
    ensemble.train_tree_models(train_data, train_targets, val_data, val_targets)
    
    # Save models
    model_dir = f"models/{symbol}/"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    ensemble.save_models(model_dir)
    
    print(f"\n✅ Training complete! Models saved to {model_dir}")
    return True


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fetch data from Polygon and train LSTM ensemble models'
    )
    parser.add_argument(
        '--fetch-only',
        action='store_true',
        help='Only fetch data, do not train models'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Only train models (assumes data already fetched)'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Symbols to process (default: from config)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=60,
        help='Days of historical data to fetch (default: 60)'
    )
    parser.add_argument(
        '--train-days',
        type=int,
        default=30,
        help='Days of data to use for training (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Get symbols
    symbols = args.symbols if args.symbols else config._config['symbols']
    
    print(f"\n{'='*60}")
    print("📊 Polygon Data Fetch & Model Training")
    print(f"{'='*60}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Days to fetch: {args.days}")
    print(f"Days for training: {args.train_days}")
    print(f"{'='*60}\n")
    
    # Fetch data
    if not args.train_only:
        print("Step 1: Fetching data from Polygon API...\n")
        
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            print("❌ POLYGON_API_KEY not found in environment!")
            return
        
        fetcher = PolygonDataFetcher(api_key)
        
        # Fetch data
        df = fetcher.fetch_multiple_symbols(symbols, days_back=args.days)
        
        if df.empty:
            print("❌ No data fetched. Cannot proceed.")
            return
        
        # Save to database
        db_connection = os.getenv('DB_CONNECTION')
        save_to_database(df, db_connection, 'market_data_minute')
        
        print(f"\n✅ Data fetched and saved!")
        print(f"   Total records: {len(df)}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Train models
    if not args.fetch_only:
        print(f"\n\nStep 2: Training models...\n")
        
        success_count = 0
        for symbol in symbols:
            try:
                if train_model_for_symbol(symbol, config, days=args.train_days):
                    success_count += 1
            except Exception as e:
                print(f"❌ Error training {symbol}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"   Successfully trained: {success_count}/{len(symbols)} symbols")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
