# Quick Start Guide - LSTM Ensemble Trading System

## 🚀 Get Started in 5 Minutes

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Polygon.io API key

### Step 1: Installation

```bash
# Extract the package
unzip lstm_ensemble_trading_system.zip
cd lstm_ensemble_trading_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

Update these values:
```
DB_CONNECTION=postgresql://your_user:your_password@localhost:5432/trading_db
POLYGON_API_KEY=your_polygon_api_key
DEVICE=cuda  # or cpu
```

### Step 3: Database Setup

```bash
# Create database (in PostgreSQL)
createdb trading_db

# Setup tables
cd scripts
python setup_database.py
```

### Step 4: Start Data Ingestion

```bash
# Terminal 1 - Data ingestion
python start_ingestion.py
```

This will stream live minute-level data from Polygon to your database.

### Step 5: Train Models (One-time)

```bash
# Terminal 2 - Wait for ~30 days of data, then train
python train_models.py --symbol AAPL --days 30
```

This takes about 30-60 minutes depending on your hardware.

### Step 6: Live Inference

```bash
# Terminal 3 - Start predictions
python run_inference.py --symbol AAPL
```

You'll see live predictions every minute!

### Step 7: Monitor Performance

```bash
# Terminal 4 - Check accuracy
python analyze_performance.py --symbol AAPL --hours 24
```

## 📊 Expected Output

### Inference Output
```
================================================================================
⏰ 2024-02-01 14:30:00 | 📈 AAPL | 💰 $185.42
--- Models ---
  lstm                :  +0.15%
  bilstm              :  +0.18%
  attention_lstm      :  +0.16%
  cnn_lstm            :  +0.12%
  lgbm                :  +0.14%
  xgb                 :  +0.13%
--- Ensemble ---
  Prediction: +0.15%
  Confidence: 67.50%
  Agreement:  85.30%
📊 Total: 1,243
================================================================================

🚨 SIGNAL: BUY | Strength: 1.85
   Price: $185.42 → Target: $185.98
   Stop Loss: $184.85 | Confidence: 67.50%
```

### Accuracy Report
```
================================================================================
📊 ACCURACY REPORT - AAPL
================================================================================
  Directional Accuracy: 57.30%
  Mean Absolute Error:  0.001234
  RMSE:                 0.002156
  Total Predictions:    1,440
================================================================================
```

## 📁 Package Structure

```
lstm_ensemble_trading_system/
├── lstm_ensemble/              # Main package
│   ├── complete_system.py     # All-in-one implementation
│   └── __init__.py
├── scripts/                    # Executable scripts
│   ├── setup_database.py
│   ├── start_ingestion.py
│   ├── train_models.py
│   ├── run_inference.py
│   └── analyze_performance.py
├── sql/                        # SQL schemas and queries
│   ├── create_tables.sql
│   └── queries.sql
├── config/                     # Configuration
│   └── config.yaml
├── models/                     # Saved models (created during training)
├── requirements.txt
├── setup.py
├── README.md
├── QUICKSTART.md
└── LICENSE
```

## 💾 Database Tables

After setup, you'll have 3 tables:

1. **market_data_minute**: Live OHLCV data from Polygon
2. **model_predictions**: All model predictions with confidence scores
3. **trading_signals**: BUY/SELL signals with risk parameters

## 📈 View Results in Database

```sql
-- Latest predictions
SELECT * FROM model_predictions 
WHERE symbol = 'AAPL' 
ORDER BY timestamp DESC LIMIT 10;

-- Active signals
SELECT * FROM trading_signals 
WHERE signal != 'HOLD' AND is_executed = FALSE
ORDER BY confidence_score DESC;

-- Accuracy check
SELECT 
    COUNT(*) as total,
    AVG(CASE WHEN SIGN(ensemble_pred) = SIGN(actual_return) 
        THEN 1 ELSE 0 END) * 100 as accuracy_pct
FROM model_predictions
WHERE timestamp >= NOW() - INTERVAL '24 hours';
```

## 🎯 Performance Expectations

**Realistic targets for minute-level trading:**
- Directional Accuracy: 55-60%
- High Confidence Accuracy: 60-65%
- Profit Factor: >1.5
- Sharpe Ratio: >1.0

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'polygon'"
```bash
pip install polygon-api-client
```

### "Database connection error"
- Verify PostgreSQL is running: `pg_isready`
- Check credentials in .env file
- Test connection: `psql -U your_user -d trading_db`

### "CUDA out of memory"
- Change DEVICE to "cpu" in .env
- Reduce batch_size in config/config.yaml

### "No data available for training"
- Wait for data ingestion to collect more data
- Check if start_ingestion.py is running
- Verify Polygon API key is valid

## 📚 Next Steps

1. **Customize features**: Edit `config/config.yaml` to add/remove features
2. **Tune ensemble weights**: Adjust weights based on your accuracy analysis
3. **Add more symbols**: Update symbols list in config
4. **Backtest**: Use the collected data to backtest strategies
5. **Deploy**: Use Docker or systemd for production deployment

## 🆘 Support

- Documentation: See README.md
- SQL Queries: See sql/queries.sql
- Configuration: See config/config.yaml

## ⚠️ Important Notes

- **Paper trading first**: Test thoroughly before live trading
- **Transaction costs**: Factor in spreads and fees
- **Risk management**: Always use stop losses
- **Position sizing**: Never risk more than 1-2% per trade

## 🎓 Learning Resources

1. Review `lstm_ensemble/complete_system.py` for implementation details
2. Check `sql/queries.sql` for analysis examples
3. Experiment with different symbols and timeframes
4. Monitor model agreement and confidence scores

---

**Ready to trade smarter with AI? Let's go! 🚀**
