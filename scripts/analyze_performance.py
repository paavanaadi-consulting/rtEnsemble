#!/usr/bin/env python3
"""Analyze prediction performance"""

import sys
sys.path.insert(0, '..')

from lstm_ensemble import Config, PredictionAccuracyAnalyzer
import argparse

def analyze(symbol: str, hours: int = 24):
    config = Config()
    
    analyzer = PredictionAccuracyAnalyzer(config.db_connection)
    metrics = analyzer.calculate_accuracy_metrics(symbol, lookback_hours=hours)
    
    if metrics:
        analyzer.print_accuracy_report(metrics, symbol)
    else:
        print(f"No data available for {symbol}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='AAPL', help='Stock symbol')
    parser.add_argument('--hours', type=int, default=24, help='Lookback hours')
    args = parser.parse_args()
    
    analyze(args.symbol, args.hours)
