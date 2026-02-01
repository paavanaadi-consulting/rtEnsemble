#!/usr/bin/env python3
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)
cur = conn.cursor()

# Check for PRSO data
print("="*60)
print("Checking PRSO:")
print("="*60)
cur.execute("SELECT COUNT(*) FROM market_data_minute WHERE symbol = 'PRSO'")
count = cur.fetchone()[0]
print(f'PRSO records: {count}')

if count > 0:
    cur.execute("""
        SELECT MIN(timestamp), MAX(timestamp) 
        FROM market_data_minute WHERE symbol = 'PRSO'
    """)
    date_range = cur.fetchone()
    print(f'Date range: {date_range[0]} to {date_range[1]}')

print()

# Check for KITT data
print("="*60)
print("Checking KITT:")
print("="*60)
cur.execute("SELECT COUNT(*) FROM market_data_minute WHERE symbol = 'KITT'")
count = cur.fetchone()[0]
print(f'KITT records: {count}')

if count > 0:
    cur.execute("""
        SELECT MIN(timestamp), MAX(timestamp) 
        FROM market_data_minute WHERE symbol = 'KITT'
    """)
    date_range = cur.fetchone()
    print(f'Date range: {date_range[0]} to {date_range[1]}')

print()

# Check all symbols in database
print("="*60)
print("All symbols in database:")
print("="*60)
cur.execute("""
    SELECT symbol, COUNT(*) as count, MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
    FROM market_data_minute 
    GROUP BY symbol 
    ORDER BY symbol
""")

results = cur.fetchall()
print(f'Total unique symbols: {len(results)}\n')
for row in results:
    print(f'{row[0]:10s} - {row[1]:,} records ({row[2]} to {row[3]})')

cur.close()
conn.close()
