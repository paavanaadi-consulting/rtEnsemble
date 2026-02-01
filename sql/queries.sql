-- Useful analysis queries

-- Latest predictions
SELECT 
    timestamp,
    symbol,
    current_price,
    ensemble_pred * 100 as predicted_return_pct,
    prediction_confidence,
    model_agreement
FROM model_predictions
WHERE symbol = 'AAPL'
ORDER BY timestamp DESC
LIMIT 20;

-- Active trading signals
SELECT 
    timestamp,
    symbol,
    signal,
    signal_strength,
    predicted_return * 100 as predicted_return_pct,
    current_price,
    target_price,
    stop_loss,
    confidence_score
FROM trading_signals
WHERE is_executed = FALSE
    AND signal IN ('BUY', 'SELL')
ORDER BY timestamp DESC;

-- Performance summary
SELECT 
    symbol,
    signal,
    COUNT(*) as total_signals,
    AVG(confidence_score) as avg_confidence,
    SUM(CASE WHEN is_executed THEN 1 ELSE 0 END) as executed_count,
    AVG(actual_return) * 100 as avg_return_pct,
    SUM(pnl) as total_pnl
FROM trading_signals
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY symbol, signal
ORDER BY total_pnl DESC;

-- Model comparison
SELECT 
    timestamp,
    symbol,
    lstm_pred,
    bilstm_pred,
    attention_lstm_pred,
    cnn_lstm_pred,
    lgbm_pred,
    xgb_pred,
    ensemble_pred,
    model_agreement
FROM model_predictions
WHERE symbol = 'AAPL'
ORDER BY timestamp DESC
LIMIT 10;

-- Accuracy check
WITH accuracy_check AS (
    SELECT 
        COUNT(*) as total_predictions,
        AVG(CASE 
            WHEN SIGN(ensemble_pred) = SIGN(
                (LEAD(m.close, 5) OVER (ORDER BY m.timestamp) - m.close) / m.close
            ) THEN 1 ELSE 0 
        END) as directional_accuracy,
        AVG(prediction_confidence) as avg_confidence
    FROM model_predictions p
    JOIN market_data_minute m 
        ON p.symbol = m.symbol AND p.timestamp = m.timestamp
    WHERE p.symbol = 'AAPL'
        AND p.timestamp >= NOW() - INTERVAL '24 hours'
)
SELECT 
    total_predictions,
    ROUND(directional_accuracy * 100, 2) as accuracy_pct,
    ROUND(avg_confidence * 100, 2) as avg_confidence_pct
FROM accuracy_check;
