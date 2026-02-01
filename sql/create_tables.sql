-- Market Data Table
CREATE TABLE IF NOT EXISTS market_data_minute (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(12, 4),
    high DECIMAL(12, 4),
    low DECIMAL(12, 4),
    close DECIMAL(12, 4),
    volume BIGINT,
    vwap DECIMAL(12, 4),
    transactions INTEGER,
    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp)
);

CREATE INDEX idx_market_data_symbol_timestamp ON market_data_minute(symbol, timestamp DESC);
CREATE INDEX idx_market_data_timestamp ON market_data_minute(timestamp DESC);

-- Predictions Table
CREATE TABLE IF NOT EXISTS model_predictions (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    current_price DECIMAL(12, 4),
    lstm_pred DECIMAL(10, 6),
    bilstm_pred DECIMAL(10, 6),
    attention_lstm_pred DECIMAL(10, 6),
    cnn_lstm_pred DECIMAL(10, 6),
    lgbm_pred DECIMAL(10, 6),
    xgb_pred DECIMAL(10, 6),
    ensemble_pred DECIMAL(10, 6),
    prediction_confidence DECIMAL(5, 4),
    model_agreement DECIMAL(5, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp)
);

CREATE INDEX idx_predictions_symbol_timestamp ON model_predictions(symbol, timestamp DESC);
CREATE INDEX idx_predictions_created ON model_predictions(created_at DESC);

-- Trading Signals Table
CREATE TABLE IF NOT EXISTS trading_signals (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    signal VARCHAR(10) NOT NULL,
    signal_strength DECIMAL(5, 4),
    predicted_return DECIMAL(10, 6),
    current_price DECIMAL(12, 4),
    target_price DECIMAL(12, 4),
    stop_loss DECIMAL(12, 4),
    confidence_score DECIMAL(5, 4),
    volatility DECIMAL(10, 6),
    is_executed BOOLEAN DEFAULT FALSE,
    execution_price DECIMAL(12, 4),
    execution_time TIMESTAMP,
    actual_return DECIMAL(10, 6),
    pnl DECIMAL(12, 4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp)
);

CREATE INDEX idx_signals_symbol_timestamp ON trading_signals(symbol, timestamp DESC);
CREATE INDEX idx_signals_signal ON trading_signals(signal, timestamp DESC);
CREATE INDEX idx_signals_executed ON trading_signals(is_executed, timestamp DESC);
