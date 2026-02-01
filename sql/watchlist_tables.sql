-- LSTM Ensemble Output Table for Watchlist Alerts
-- ================================================
-- Note: Assumes public.watchlist_alerts table already exists
-- This table maintains 1 record per symbol with the latest prediction

-- Table: public.lstm_ensemble_output
-- This table stores the LSTM ensemble predictions for each watchlist alert
CREATE TABLE IF NOT EXISTS public.lstm_ensemble_output (
    symbol VARCHAR(10) PRIMARY KEY,
    watchlist_alert_id BIGINT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    alert_type VARCHAR(50),
    current_price DECIMAL(12, 4),
    
    -- Individual model predictions (fractional returns)
    lstm_pred DECIMAL(10, 6),
    bilstm_pred DECIMAL(10, 6),
    attention_lstm_pred DECIMAL(10, 6),
    cnn_lstm_pred DECIMAL(10, 6),
    lgbm_pred DECIMAL(10, 6),
    xgb_pred DECIMAL(10, 6),
    
    -- Ensemble prediction
    ensemble_pred DECIMAL(10, 6),  -- Fractional return prediction
    predicted_return_pct DECIMAL(10, 4),  -- Return as percentage
    predicted_price DECIMAL(12, 4),  -- Predicted future price
    
    -- Confidence metrics
    prediction_confidence DECIMAL(5, 4),  -- 0 to 1
    model_agreement DECIMAL(5, 4),  -- 0 to 1
    
    -- Trading signal
    signal VARCHAR(10) CHECK (signal IN ('BUY', 'SELL', 'HOLD')),
    signal_strength DECIMAL(5, 4),  -- 0 to 1
    target_price DECIMAL(12, 4),
    stop_loss DECIMAL(12, 4),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for lstm_ensemble_output
CREATE INDEX IF NOT EXISTS idx_lstm_output_signal 
    ON public.lstm_ensemble_output(signal);
CREATE INDEX IF NOT EXISTS idx_lstm_output_updated 
    ON public.lstm_ensemble_output(updated_at DESC);

-- View: Latest Predictions with Strong Signals
CREATE OR REPLACE VIEW public.v_strong_signals AS
SELECT 
    e.symbol,
    e.timestamp,
    e.watchlist_alert_id,
    e.current_price,
    e.ensemble_pred,
    e.predicted_return_pct,
    e.predicted_price,
    e.prediction_confidence,
    e.model_agreement,
    e.signal,
    e.signal_strength,
    e.target_price,
    e.stop_loss,
    e.updated_at,
    -- Potential profit/loss
    CASE 
        WHEN e.signal = 'BUY' THEN (e.target_price - e.current_price) / e.current_price * 100
        WHEN e.signal = 'SELL' THEN (e.current_price - e.target_price) / e.current_price * 100
        ELSE 0
    END as potential_return_pct
FROM public.lstm_ensemble_output e
WHERE e.signal != 'HOLD'
    AND e.prediction_confidence >= 0.6
    AND e.signal_strength >= 0.5
ORDER BY e.signal_strength DESC, e.updated_at DESC;


-- View: All Current Predictions Summary
CREATE OR REPLACE VIEW public.v_current_predictions AS
SELECT 
    symbol,
    current_price,
    ensemble_pred,
    predicted_return_pct,
    predicted_price,
    prediction_confidence,
    model_agreement,
    signal,
    signal_strength,
    target_price,
    stop_loss,
    timestamp,
    updated_at,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - updated_at))/60 as minutes_since_update
FROM public.lstm_ensemble_output
ORDER BY updated_at DESC;


-- Function: Get symbols needing update (no prediction in last N minutes)
CREATE OR REPLACE FUNCTION public.get_stale_predictions(minutes_threshold INTEGER DEFAULT 60)
RETURNS TABLE(symbol VARCHAR, last_update TIMESTAMP, minutes_old NUMERIC) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.symbol,
        e.updated_at,
        EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - e.updated_at))/60 as minutes_old
    FROM public.lstm_ensemble_output e
    WHERE EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - e.updated_at))/60 > minutes_threshold
    ORDER BY e.updated_at ASC;
END;
$$ LANGUAGE plpgsql;


-- Function: Get prediction summary for a symbol
CREATE OR REPLACE FUNCTION public.get_symbol_prediction(p_symbol VARCHAR)
RETURNS TABLE(
    symbol VARCHAR,
    current_price DECIMAL,
    predicted_price DECIMAL,
    predicted_return_pct DECIMAL,
    signal VARCHAR,
    confidence DECIMAL,
    updated_at TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.symbol,
        e.current_price,
        e.predicted_price,
        e.predicted_return_pct,
        e.signal,
        e.prediction_confidence,
        e.updated_at
    FROM public.lstm_ensemble_output e
    WHERE e.symbol = p_symbol;
END;
$$ LANGUAGE plpgsql;


-- Sample queries
COMMENT ON TABLE public.lstm_ensemble_output IS 
'Stores latest LSTM ensemble predictions for each symbol (1 record per symbol)';

COMMENT ON VIEW public.v_strong_signals IS 
'Shows predictions with strong trading signals (confidence >= 0.6, strength >= 0.5)';

COMMENT ON VIEW public.v_current_predictions IS 
'Current predictions for all symbols with time since last update';

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE ON public.lstm_ensemble_output TO your_app_user;
-- GRANT SELECT ON public.v_strong_signals TO your_app_user;
-- GRANT SELECT ON public.v_current_predictions TO your_app_user;
