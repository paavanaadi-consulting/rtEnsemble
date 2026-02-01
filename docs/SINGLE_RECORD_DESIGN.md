# LSTM Ensemble Output - Single Record Per Symbol Design

## Overview

The `public.lstm_ensemble_output` table now maintains **1 record per symbol** with INSERT/UPDATE logic. Each time a new prediction is made for a symbol, the existing record is updated with the latest data.

## Table Structure

```sql
CREATE TABLE public.lstm_ensemble_output (
    symbol VARCHAR(10) PRIMARY KEY,  -- One record per symbol
    watchlist_alert_id BIGINT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    current_price DECIMAL(12, 4),
    
    -- Model predictions
    lstm_pred DECIMAL(10, 6),
    bilstm_pred DECIMAL(10, 6),
    attention_lstm_pred DECIMAL(10, 6),
    cnn_lstm_pred DECIMAL(10, 6),
    lgbm_pred DECIMAL(10, 6),
    xgb_pred DECIMAL(10, 6),
    
    -- Ensemble results
    ensemble_pred DECIMAL(10, 6),
    predicted_return_pct DECIMAL(10, 4),
    predicted_price DECIMAL(12, 4),
    prediction_confidence DECIMAL(5, 4),
    model_agreement DECIMAL(5, 4),
    
    -- Trading signal
    signal VARCHAR(10),  -- BUY/SELL/HOLD
    signal_strength DECIMAL(5, 4),
    target_price DECIMAL(12, 4),
    stop_loss DECIMAL(12, 4),
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- First prediction
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP   -- Latest prediction
);
```

## Key Changes

### 1. Primary Key: `symbol` instead of `id`
- **Before**: Auto-incrementing `id` with multiple records per symbol
- **After**: `symbol` as primary key, ensuring only 1 record per symbol

### 2. INSERT ... ON CONFLICT UPDATE
```sql
INSERT INTO public.lstm_ensemble_output (...)
VALUES (...)
ON CONFLICT (symbol) DO UPDATE SET
    -- Updates all fields with latest values
    watchlist_alert_id = EXCLUDED.watchlist_alert_id,
    timestamp = EXCLUDED.timestamp,
    ...
    updated_at = CURRENT_TIMESTAMP
```

### 3. Timestamp Tracking
- `created_at`: When symbol was first added (never changes)
- `updated_at`: When last prediction was made (updated on each new prediction)

## Benefits

### ✅ Always Current Data
- Each symbol shows the most recent prediction
- No need to query for "latest" record
- Simple `SELECT * FROM lstm_ensemble_output WHERE symbol = 'AAPL'`

### ✅ Space Efficient
- Maximum of N records (where N = number of unique symbols)
- No historical data accumulation
- Keeps database lean and fast

### ✅ Real-time Dashboard Ready
```sql
-- Get all current predictions
SELECT * FROM public.v_current_predictions;

-- Get strong signals
SELECT * FROM public.v_strong_signals;

-- Check if predictions are stale
SELECT * FROM public.get_stale_predictions(60);  -- Older than 60 minutes
```

## Usage Examples

### Query Current Predictions

```sql
-- All current predictions
SELECT symbol, signal, predicted_return_pct, prediction_confidence, updated_at
FROM public.lstm_ensemble_output
ORDER BY signal_strength DESC;

-- Specific symbol
SELECT * FROM public.lstm_ensemble_output 
WHERE symbol = 'AAPL';

-- Strong BUY signals
SELECT * FROM public.v_strong_signals
WHERE signal = 'BUY'
ORDER BY signal_strength DESC;

-- Recent updates
SELECT symbol, signal, updated_at,
       EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - updated_at))/60 as minutes_ago
FROM public.lstm_ensemble_output
ORDER BY updated_at DESC;
```

### Check Freshness

```sql
-- Symbols not updated in last hour
SELECT * FROM public.get_stale_predictions(60);

-- Time since last update for each symbol
SELECT 
    symbol,
    signal,
    updated_at,
    CURRENT_TIMESTAMP - updated_at as time_since_update,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - updated_at))/60 as minutes_old
FROM public.lstm_ensemble_output
ORDER BY updated_at ASC;
```

### Dashboard Queries

```sql
-- Summary statistics
SELECT 
    COUNT(*) as total_symbols,
    COUNT(CASE WHEN signal = 'BUY' THEN 1 END) as buy_signals,
    COUNT(CASE WHEN signal = 'SELL' THEN 1 END) as sell_signals,
    COUNT(CASE WHEN signal = 'HOLD' THEN 1 END) as hold_signals,
    AVG(prediction_confidence) as avg_confidence,
    MAX(updated_at) as last_update
FROM public.lstm_ensemble_output;

-- Top opportunities by predicted return
SELECT 
    symbol,
    signal,
    predicted_return_pct,
    signal_strength,
    prediction_confidence,
    current_price,
    target_price,
    updated_at
FROM public.lstm_ensemble_output
WHERE signal IN ('BUY', 'SELL')
    AND prediction_confidence > 0.65
ORDER BY ABS(predicted_return_pct) DESC
LIMIT 10;
```

## Python Usage

The script automatically handles the INSERT/UPDATE logic:

```python
# Continuous mode - updates predictions as alerts come in
python scripts/watchlist_inference.py --mode continuous --interval 30

# Single pass - updates all pending alerts once
python scripts/watchlist_inference.py --mode once
```

### What Happens

1. New alert arrives for symbol "AAPL"
2. Script processes alert and generates prediction
3. **First time**: Record inserted with `created_at` = now
4. **Subsequent times**: Record updated, `updated_at` = now, `created_at` unchanged
5. Result: Always have latest prediction for each symbol

## Monitoring

### Check Last Update Time
```python
import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine(os.getenv('DB_CONNECTION'))

# Get symbols with stale predictions
query = text("""
    SELECT symbol, updated_at,
           EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - updated_at))/60 as minutes_old
    FROM public.lstm_ensemble_output
    WHERE EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - updated_at))/60 > 60
    ORDER BY updated_at ASC
""")

df = pd.read_sql(query, engine)
print(df)
```

### Track Update Frequency
```sql
-- See how often each symbol gets updated
SELECT 
    symbol,
    updated_at,
    LAG(updated_at) OVER (PARTITION BY symbol ORDER BY updated_at) as previous_update,
    updated_at - LAG(updated_at) OVER (PARTITION BY symbol ORDER BY updated_at) as update_interval
FROM public.lstm_ensemble_output
ORDER BY symbol, updated_at DESC;
```

## Migration from Old Schema

If you have existing data with multiple records per symbol:

```sql
-- Create new table with single-record-per-symbol structure
-- (Already handled by the script)

-- Migrate latest prediction for each symbol
INSERT INTO public.lstm_ensemble_output_new
SELECT DISTINCT ON (symbol)
    symbol, watchlist_alert_id, timestamp, 
    -- ... all other columns
FROM public.lstm_ensemble_output_old
ORDER BY symbol, timestamp DESC;

-- Rename tables
DROP TABLE public.lstm_ensemble_output_old;
ALTER TABLE public.lstm_ensemble_output_new RENAME TO lstm_ensemble_output;
```

## Comparison: Before vs After

| Aspect | Before (Multiple Records) | After (Single Record) |
|--------|---------------------------|----------------------|
| **Records** | Many per symbol | 1 per symbol |
| **Query for latest** | `WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1` | `WHERE symbol = ?` |
| **Space usage** | Grows indefinitely | Fixed (N symbols) |
| **Updates** | Always INSERT | INSERT or UPDATE |
| **Dashboard** | Need aggregation | Direct SELECT |
| **History** | Maintained | Not maintained |

## When to Use This Design

✅ **Good For:**
- Real-time dashboards showing current state
- Trading systems needing latest signals
- Limited storage/memory
- Fast queries for current state

❌ **Not Good For:**
- Historical analysis of predictions
- Tracking prediction accuracy over time
- Audit trails
- Time-series analysis

## Adding Historical Tracking (Optional)

If you need history, create a separate archive table:

```sql
-- Archive table for historical predictions
CREATE TABLE public.lstm_ensemble_history (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    prediction_time TIMESTAMP,
    -- ... all prediction columns ...
);

-- Trigger to archive before update
CREATE TRIGGER archive_prediction
BEFORE UPDATE ON public.lstm_ensemble_output
FOR EACH ROW
EXECUTE FUNCTION archive_old_prediction();
```

---

**Version:** 2.0.0  
**Date:** February 1, 2026  
**Change:** Switched from multi-record to single-record-per-symbol design
