# LSTM Ensemble Trading System

A production-ready PyTorch-based ensemble system for minute-level stock trading predictions.

## Features

- ✅ Multiple LSTM architectures (Vanilla, Bidirectional, Attention, CNN-LSTM Hybrid)
- ✅ Tree-based models (LightGBM, XGBoost) for ensemble diversity
- ✅ Real-time data ingestion from Polygon.io
- ✅ Live inference with database output
- ✅ Comprehensive accuracy tracking and performance analysis
- ✅ Trading signal generation with risk management
- ✅ Modular, scalable architecture

## Quick Start

### 1. Installation

```bash
# Clone/extract the package
cd lstm_ensemble_trading_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env  # Or use your preferred editor
```

### 3. Database Setup

```bash
# Create database and tables
python scripts/setup_database.py
```

### 4. Data Ingestion (Terminal 1)

```bash
# Start streaming Polygon data
python scripts/start_ingestion.py
```

### 5. Model Training (One-time or periodic)

```bash
# Train the ensemble models
python scripts/train_models.py --symbol AAPL --days 30
```

### 6. Live Inference (Terminal 2)

```bash
# Start live predictions
python scripts/run_inference.py --symbol AAPL
```

### 7. Performance Analysis

```bash
# Analyze prediction accuracy
python scripts/analyze_performance.py --symbol AAPL --hours 24
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     POLYGON REAL-TIME STREAM                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │  DATA INGESTION ENGINE │
                │   (WebSocket → DB)     │
                └────────┬───────────────┘
                         │
                         ▼
                ┌────────────────────────┐
                │  MARKET_DATA_MINUTE    │
                │      (PostgreSQL)      │
                └────────┬───────────────┘
                         │
                         ▼
                ┌────────────────────────┐
                │  FEATURE ENGINEERING   │
                │   (25+ Indicators)     │
                └────────┬───────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   ┌────────┐      ┌────────┐       ┌────────┐
   │  LSTM  │      │ BiLSTM │       │  LGBM  │
   │ Models │      │ Models │       │  XGB   │
   └───┬────┘      └───┬────┘       └───┬────┘
       │               │                │
       └───────────────┼────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │    ENSEMBLE    │
              │   (Weighted)   │
              └────────┬───────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
  ┌──────────┐  ┌───────────┐  ┌──────────┐
  │PREDICTIONS│  │  SIGNALS  │  │ ANALYSIS │
  │  TABLE   │  │   TABLE   │  │   LOGS   │
  └──────────┘  └───────────┘  └──────────┘
```

## Configuration

Edit `config/config.yaml`:

```yaml
database:
  connection_string: "postgresql://user:pass@localhost:5432/trading_db"

polygon:
  api_key: "YOUR_API_KEY"
  feed: "delayed"  # or "realtime"

symbols:
  - AAPL
  - TSLA
  - NVDA

models:
  sequence_length: 60  # minutes lookback
  prediction_horizon: 5  # minutes ahead
  device: "cuda"  # or "cpu"
```

## Project Structure

```
lstm_ensemble_trading_system/
├── lstm_ensemble/           # Main package
│   ├── models/             # LSTM and tree models
│   ├── training/           # Training logic
│   ├── inference/          # Live inference
│   ├── output/             # Database writers
│   ├── analysis/           # Performance analysis
│   └── utils/              # Utilities
├── scripts/                # Executable scripts
├── config/                 # Configuration files
├── sql/                    # Database schemas
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks
└── docs/                   # Documentation
```

## Models

### LSTM Architectures

1. **Vanilla LSTM**: Standard LSTM with 2 layers
2. **Bidirectional LSTM**: Processes sequences in both directions
3. **Attention LSTM**: Attention mechanism for important time steps
4. **CNN-LSTM Hybrid**: Combines convolutional and recurrent layers

### Tree Models

5. **LightGBM**: Gradient boosting for non-linear patterns
6. **XGBoost**: Extreme gradient boosting

### Ensemble

Weighted combination with optimized weights:
- LSTM: 25%
- BiLSTM: 25%
- Attention: 20%
- CNN-LSTM: 15%
- LGBM: 10%
- XGB: 5%

## Performance Expectations

### Realistic Accuracy Targets

- **Directional Accuracy**: 55-60% (overall)
- **High Confidence**: 60-65% accuracy
- **Profit Factor**: >1.5
- **Sharpe Ratio**: >1.0

### Why Minute-Level is Challenging

- High noise-to-signal ratio (~80% noise)
- Market efficiency ~90%
- Transaction costs impact
- Requires >53% accuracy to be profitable

## Database Schema

### market_data_minute
- Stores OHLCV data from Polygon
- Indexed for fast time-series queries

### model_predictions
- All model predictions
- Confidence scores
- Model agreement metrics

### trading_signals
- BUY/SELL/HOLD signals
- Risk parameters (stop loss, targets)
- Execution tracking
- P&L tracking

## API Reference

### Training

```python
from lstm_ensemble import LSTMEnsemble, Config

config = Config()
ensemble = LSTMEnsemble(
    feature_columns=config.features,
    sequence_length=60
)

# Train models
ensemble.train_pytorch_models(train_data, train_targets, val_data, val_targets)
ensemble.train_tree_models(train_data, train_targets, val_data, val_targets)

# Save
ensemble.save_models("models/")
```

### Inference

```python
from lstm_ensemble import LiveInferenceEngine

engine = LiveInferenceEngine(
    ensemble=ensemble,
    data_reader=data_reader,
    feature_columns=config.features,
    db_connection_string=config.database.connection_string
)

engine.start_live_inference(interval_seconds=60)
```

### Analysis

```python
from lstm_ensemble import PredictionAccuracyAnalyzer

analyzer = PredictionAccuracyAnalyzer(db_connection)
metrics = analyzer.calculate_accuracy_metrics('AAPL', lookback_hours=24)
analyzer.print_accuracy_report(metrics, 'AAPL')
analyzer.plot_accuracy_analysis('accuracy_plot.png')
```

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=lstm_ensemble --cov-report=html
```

## Deployment

### Docker Deployment

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f inference
```

### Production Checklist

- [ ] Configure proper database credentials
- [ ] Set up Polygon API key (realtime subscription)
- [ ] Configure logging and monitoring
- [ ] Set up backup strategy for models
- [ ] Implement alerting for system failures
- [ ] Configure auto-restart on crashes
- [ ] Set up performance monitoring dashboard

## Monitoring

### Real-time Metrics

```bash
# Watch live accuracy
python scripts/monitor_accuracy.py --symbol AAPL
```

### SQL Queries

```sql
-- Latest predictions
SELECT * FROM model_predictions 
WHERE symbol = 'AAPL' 
ORDER BY timestamp DESC LIMIT 10;

-- Active signals
SELECT * FROM trading_signals 
WHERE is_executed = FALSE AND signal != 'HOLD'
ORDER BY confidence_score DESC;

-- Performance summary
SELECT 
    symbol,
    AVG(CASE WHEN SIGN(ensemble_pred) = SIGN(actual_return) THEN 1 ELSE 0 END) as accuracy
FROM model_predictions
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY symbol;
```

## Troubleshooting

### Common Issues

**Q: Low accuracy (<52%)**
- Check data quality
- Verify feature engineering
- Ensure sufficient training data
- Review ensemble weights

**Q: Database connection errors**
- Verify credentials in .env
- Check PostgreSQL is running
- Test connection with psql

**Q: Polygon stream disconnects**
- Check API key validity
- Verify subscription level
- Monitor rate limits

**Q: Out of memory errors**
- Reduce batch size
- Use CPU instead of GPU for small datasets
- Increase system memory

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details

## Support

- Documentation: `docs/`
- Issues: GitHub Issues
- Email: your.email@example.com

## Acknowledgments

- PyTorch team for excellent deep learning framework
- Polygon.io for reliable market data
- LightGBM and XGBoost teams

## Citation

If you use this in research, please cite:

```bibtex
@software{lstm_ensemble_trading,
  title={LSTM Ensemble Trading System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/lstm-ensemble-trading}
}
```

## Changelog

### v1.0.0 (2024-02-01)
- Initial release
- Complete LSTM ensemble implementation
- Real-time inference engine
- Comprehensive accuracy tracking
- Production-ready deployment

---

**Note**: This system is for educational and research purposes. Always do your own research and never invest more than you can afford to lose.
