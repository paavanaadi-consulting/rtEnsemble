#!/usr/bin/env python3
"""Start Polygon data ingestion"""

import sys
sys.path.insert(0, '..')

from lstm_ensemble.complete_system import PolygonToDatabase
from lstm_ensemble import Config

def start_ingestion():
    config = Config()
    
    ingestion = PolygonToDatabase(
        api_key=config._config['polygon']['api_key'],
        db_connection_string=config.db_connection,
        table_name=config.market_data_table,
        symbols=config.symbols,
        batch_size=100
    )
    
    print("Starting Polygon data ingestion...")
    ingestion.start_streaming()

if __name__ == "__main__":
    start_ingestion()
