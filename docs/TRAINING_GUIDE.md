# Training Models from Polygon API

## Quick Start Guide

### 1. Fetch Data and Train Models

```bash
# Train single symbol
python scripts/fetch_and_train.py --symbols AAPL --days 60 --train-days 30

# Train multiple symbols (from config)
python scripts/fetch_and_train.py --days 60 --train-days 30

# Train specific symbols
python scripts/fetch_and_train.py --symbols AAPL TSLA NVDA --days 90 --train-days 45
```

### 2. Fetch Data Only

```bash
python scripts/fetch_and_train.py --fetch-only --days 60
```

### 3. Train Only (if data already exists)

```bash
python scripts/fetch_and_train.py --train-only --train-days 30
```

## Process Overview

### Data Fetching
1. **Source**: Polygon.io REST API
2. **Data Type**: 1-minute aggregated bars
3. **Storage**: PostgreSQL `market_data_minute` table
4. **Rate Limits**: 5 requests/minute (free tier) - script handles automatically

### Model Training
1. **Data Loading**: Loads from `market_data_minute` table
2. **Feature Engineering**: Computes 26+ technical indicators
3. **Target Creation**: 5-minute ahead price return
4. **Train/Val Split**: 80/20 split
5. **Models Trained**:
   - LSTM (25% weight)
   - Bidirectional LSTM (25% weight)
   - Attention LSTM (20% weight)
   - CNN-LSTM Hybrid (15% weight)
   - LightGBM (10% weight)
   - XGBoost (5% weight)

## Training Progress

Training typically takes:
- **AAPL** (15K records): ~5-10 minutes
- **Per epoch**: ~10-30 seconds
- **Total epochs**: 50 (with early stopping)

### What's Happening

```
Training lstm...
Epoch 1/50: train_loss=0.0234, val_loss=0.0189 ✓ [best]
Epoch 2/50: train_loss=0.0198, val_loss=0.0175 ✓ [best]
...
```

Models train until:
- Validation loss stops improving for 10 epochs
- Maximum 50 epochs reached

## Output Structure

```
models/
  ├── AAPL/
  │   ├── lstm_model.pth           # LSTM model
  │   ├── bilstm_model.pth         # BiLSTM model
  │   ├── attention_lstm_model.pth # Attention LSTM
  │   ├── cnn_lstm_model.pth       # CNN-LSTM hybrid
  │   ├── lgbm_model.pkl           # LightGBM
  │   ├── xgb_model.pkl            # XGBoost
  │   └── scaler.pkl               # Feature scaler
  ├── TSLA/
  │   └── ...
  └── NVDA/
      └── ...
```

## Data Requirements

### Minimum Data
- **Sequence Length**: 60 minutes
- **Minimum Records**: 500 bars (after features)
- **Recommended**: 10,000+ bars for good training

### Data Quality
- No missing values in OHLCV
- Continuous time series (no large gaps)
- At least 100 records for validation set

## Configuration

Edit `config/config.yaml`:

```yaml
symbols:
  - AAPL
  - TSLA
  - NVDA
  - GOOGL
  - MSFT

models:
  sequence_length: 60
  prediction_horizon: 5
  device: "cpu"  # or "cuda" for GPU

training:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  validation_split: 0.2
  early_stopping_patience: 10

features:
  - returns
  - log_returns
  - rsi
  - macd
  # ... 26 features total
```

## Checking Training Status

### Database Check
```sql
-- Check fetched data
SELECT symbol, COUNT(*) as bars, 
       MIN(timestamp) as first, MAX(timestamp) as last
FROM market_data_minute
GROUP BY symbol
ORDER BY symbol;

-- Check trained models
SELECT symbol FROM lstm_ensemble_output;
```

### File System Check
```bash
# List trained models
ls -la models/

# Check AAPL models
ls -la models/AAPL/
```

### Test Predictions
```bash
# Run inference after training
python scripts/watchlist_inference.py --mode once
```

## Troubleshooting

### Issue: "No data returned"
**Cause**: Symbol not available or date range invalid  
**Solution**: Check symbol ticker, try shorter date range

### Issue: "Rate limit hit"
**Cause**: Too many API requests  
**Solution**: Script auto-waits 60 seconds, or upgrade Polygon plan

### Issue: "Insufficient data"
**Cause**: Not enough bars for training  
**Solution**: Increase `--days` parameter (e.g., 90 or 120)

### Issue: "CUDA out of memory"
**Cause**: GPU memory insufficient  
**Solution**: Set `device: "cpu"` in config or reduce `batch_size`

### Issue: "KeyError: feature name"
**Cause**: Feature engineering failed  
**Solution**: Check data has all OHLCV columns, no NaN values

## Training Multiple Symbols

### Sequential (Recommended for Free Tier)
```bash
# Train one at a time
python scripts/fetch_and_train.py --symbols AAPL --days 60
python scripts/fetch_and_train.py --symbols TSLA --days 60
python scripts/fetch_and_train.py --symbols NVDA --days 60
```

### Batch (Requires Higher Tier)
```bash
# Train all at once
python scripts/fetch_and_train.py --symbols AAPL TSLA NVDA GOOGL MSFT --days 60
```

## After Training

### 1. Verify Models Exist
```bash
ls -la models/AAPL/
# Should show 7 files: 6 models + 1 scaler
```

### 2. Run Inference
```bash
# Single pass
python scripts/watchlist_inference.py --mode once

# Continuous monitoring
python scripts/watchlist_inference.py --mode continuous --interval 30
```

### 3. Check Predictions
```sql
SELECT * FROM public.lstm_ensemble_output;
```

## Performance Metrics

During training, you'll see:
```
Training lstm...
Epoch 1/50: train_loss=0.0234, val_loss=0.0189 ✓ [best]
Epoch 5/50: train_loss=0.0156, val_loss=0.0145 ✓ [best]
Epoch 10/50: train_loss=0.0134, val_loss=0.0132 ✓ [best]
...
Early stopping: val_loss has not improved for 10 epochs
Final val_loss: 0.0128
```

**Good signs:**
- Validation loss decreasing
- Train/Val loss close together (not overfitting)
- Final val_loss < 0.02

**Warning signs:**
- Train loss << Val loss (overfitting)
- Val loss increasing (diverging)
- Very high losses (> 0.05)

## Advanced Options

### GPU Training
```yaml
# config.yaml
models:
  device: "cuda"  # Use GPU

training:
  batch_size: 128  # Larger batches with GPU
```

### Longer Training
```yaml
training:
  epochs: 100  # More epochs
  early_stopping_patience: 20  # More patience
```

### More Data
```bash
# Fetch 6 months of data
python scripts/fetch_and_train.py --days 180 --train-days 90
```

## Next Steps

After training completes:

1. ✅ **Verify models** - Check `models/` directory
2. ✅ **Test inference** - Run `watchlist_inference.py --mode once`
3. ✅ **Check predictions** - Query `lstm_ensemble_output` table
4. ✅ **Monitor accuracy** - Compare predictions vs actual prices
5. ✅ **Deploy** - Set up continuous monitoring

---

**Estimated Training Time**: 5-15 minutes per symbol  
**Data Fetched**: ~28,000 bars per symbol (60 days)  
**Models Generated**: 6 models + 1 scaler per symbol  
**Total Size**: ~50-100 MB per symbol
