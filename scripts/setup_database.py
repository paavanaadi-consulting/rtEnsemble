#!/usr/bin/env python3
"""Setup database tables for LSTM ensemble system"""

import sys
sys.path.insert(0, '..')

from lstm_ensemble import Config
from sqlalchemy import create_engine, text

def setup_database():
    config = Config()
    engine = create_engine(config.db_connection)
    
    print("Creating database tables...")
    
    # Read SQL from file
    with open('../sql/create_tables.sql', 'r') as f:
        sql = f.read()
    
    with engine.connect() as conn:
        for statement in sql.split(';'):
            if statement.strip():
                conn.execute(text(statement))
        conn.commit()
    
    print("✓ Database setup complete!")

if __name__ == "__main__":
    setup_database()
