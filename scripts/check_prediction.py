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

# Get the AAPL prediction
cur.execute('''
    SELECT symbol, timestamp, current_price, ensemble_pred, 
           predicted_return_pct, predicted_price, 
           prediction_confidence, signal, signal_strength, 
           model_agreement, updated_at
    FROM lstm_ensemble_output 
    WHERE symbol = %s
''', ('AAPL',))

result = cur.fetchone()

if result:
    print('\n' + '='*60)
    print('📊 AAPL Prediction in Database')
    print('='*60)
    print('Symbol:              {}'.format(result[0]))
    print('Timestamp:           {}'.format(result[1]))
    print('Current Price:       ${:.2f}'.format(float(result[2])))
    print('Ensemble Prediction: {:.6f}'.format(float(result[3])))
    print('Return %:            {:.4f}%'.format(float(result[4])))
    print('Predicted Price:     ${:.2f}'.format(float(result[5])))
    print('Confidence:          {:.2f}%'.format(float(result[6])))
    print('Signal:              {}'.format(result[7]))
    print('Signal Strength:     {:.2f}'.format(float(result[8])))
    print('Model Agreement:     {:.6f} (lower is better)'.format(float(result[9])))
    print('Updated At:          {}'.format(result[10]))
    print('='*60 + '\n')
else:
    print('No prediction found for AAPL')

cur.close()
conn.close()
