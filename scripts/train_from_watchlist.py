#!/usr/bin/env python3
"""
Train LSTM Ensemble from Watchlist Alerts
==========================================
Reads alert data from public.watchlist_alerts, extracts features from
the JSONB details, builds 3-alert sequences, and trains the ensemble
to predict forward price returns for small-cap stocks.

Usage:
    python scripts/train_from_watchlist.py
    python scripts/train_from_watchlist.py --min-alerts 5 --epochs 100
    python scripts/train_from_watchlist.py --seq-len 3 --forward 3
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

load_dotenv()

from lstm_ensemble.watchlist_features import (
    build_watchlist_dataframe,
    build_sequences,
    WATCHLIST_FEATURE_COLUMNS,
)
from lstm_ensemble.watchlist_ensemble import WatchlistEnsemble


def load_watchlist_alerts(db_connection: str, min_alerts_per_symbol: int = 5) -> pd.DataFrame:
    """Load watchlist alerts from database."""
    engine = create_engine(db_connection)

    print("Loading watchlist alerts from database...")

    query = text("""
        SELECT
            wa.id,
            wa.symbol,
            wa.alert_type,
            wa.price,
            wa.volume,
            wa.details,
            wa.timestamp
        FROM public.watchlist_alerts wa
        WHERE wa.price IS NOT NULL
            AND wa.price::float > 0
            AND wa.symbol IN (
                SELECT symbol
                FROM public.watchlist_alerts
                WHERE price IS NOT NULL AND price::float > 0
                GROUP BY symbol
                HAVING COUNT(*) >= :min_alerts
            )
        ORDER BY wa.symbol, wa.timestamp ASC
    """)

    df = pd.read_sql(query, engine, params={"min_alerts": min_alerts_per_symbol})
    print(f"Loaded {len(df)} alerts for {df['symbol'].nunique()} symbols")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Train LSTM ensemble from watchlist alerts"
    )
    parser.add_argument(
        "--min-alerts", type=int, default=10,
        help="Minimum alerts per symbol to include (default: 10)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=3,
        help="Number of past alerts per sequence (default: 3)"
    )
    parser.add_argument(
        "--forward", type=int, default=3,
        help="Number of alerts ahead to measure return (default: 3)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--hidden", type=int, default=64,
        help="Hidden size for models (default: 64)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/watchlist",
        help="Directory to save models (default: models/watchlist)"
    )

    args = parser.parse_args()

    # Database connection
    db_connection = os.getenv("DB_CONNECTION")
    if not db_connection:
        print("DB_CONNECTION environment variable not set!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("WATCHLIST ALERT LSTM ENSEMBLE TRAINING")
    print(f"{'='*60}")
    print(f"Sequence length: {args.seq_len} alerts")
    print(f"Forward horizon: {args.forward} alerts")
    print(f"Epochs: {args.epochs}")
    print(f"Hidden size: {args.hidden}")
    print(f"Features: {len(WATCHLIST_FEATURE_COLUMNS)}")
    print(f"{'='*60}\n")

    # Step 1: Load data
    alerts_df = load_watchlist_alerts(db_connection, args.min_alerts)
    if alerts_df.empty:
        print("No alerts found. Exiting.")
        sys.exit(1)

    # Step 2: Extract features
    print("\nExtracting features from alert details...")
    feature_df = build_watchlist_dataframe(alerts_df)
    print(f"Feature matrix: {feature_df.shape}")

    # Show feature stats
    print(f"\nFeature statistics:")
    for col in WATCHLIST_FEATURE_COLUMNS[:5]:
        vals = feature_df[col]
        print(f"  {col:30s}: mean={vals.mean():.4f} std={vals.std():.4f} "
              f"min={vals.min():.4f} max={vals.max():.4f}")
    print(f"  ... ({len(WATCHLIST_FEATURE_COLUMNS)} features total)")

    # Step 3: Build sequences
    print(f"\nBuilding {args.seq_len}-alert sequences with {args.forward}-alert forward labels...")
    X, y, meta = build_sequences(feature_df, seq_len=args.seq_len, forward_alerts=args.forward)

    if len(X) == 0:
        print("No valid sequences could be built. Need more data.")
        sys.exit(1)

    print(f"Total sequences: {X.shape[0]}")
    print(f"Sequence shape: {X.shape}")
    print(f"Label distribution:")
    print(f"  Mean return: {y.mean()*100:+.3f}%")
    print(f"  Std return:  {y.std()*100:.3f}%")
    print(f"  Positive:    {(y > 0).sum()} ({(y > 0).mean()*100:.1f}%)")
    print(f"  Negative:    {(y < 0).sum()} ({(y < 0).mean()*100:.1f}%)")
    print(f"  Zero:        {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")

    # Step 4: Initialize and train ensemble
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}\n")

    device = os.getenv("DEVICE", "cpu")
    ensemble = WatchlistEnsemble(
        seq_len=args.seq_len,
        hidden_size=args.hidden,
        device=device,
    )

    ensemble.train_all(
        X, y,
        val_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Step 5: Save models
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ensemble.save_models(args.output_dir)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Models saved to: {args.output_dir}")
    print(f"Total training samples: {X.shape[0]}")
    print(f"Symbols covered: {len(set(m['symbol'] for m in meta))}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
