# Example Usage

## Quick Example

```python
from lstm_ensemble import Config, LSTMEnsemble, LiveInferenceEngine

# Load configuration
config = Config()

# Initialize ensemble
ensemble = LSTMEnsemble(
    feature_columns=config.features,
    sequence_length=60,
    device='cuda'
)

# Load trained models
ensemble.load_models('models/AAPL/')

# Start live inference
# (implementation details in scripts/run_inference.py)
```

## Training Example

```python
# Load your data
df = load_data_from_database()

# Feature engineering
from lstm_ensemble import FeatureEngineering
df = FeatureEngineering.compute_all_features(df)

# Create target
df['target'] = df['close'].pct_change(5).shift(-5)

# Train
ensemble.train_pytorch_models(train_data, train_targets, val_data, val_targets)
ensemble.save_models('models/AAPL/')
```
