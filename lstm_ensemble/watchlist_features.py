"""
Watchlist Alert Feature Extraction
====================================
Extracts numerical features from watchlist_alerts JSONB details
for training LSTM models on small-cap alert sequences.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional


# Alert type encoding (ordinal - higher = stronger signal)
ALERT_TYPE_MAP = {
    "INFO": 0,
    "WARNING": 1,
    "WATCH": 2,
    "ALERT": 3,
    "STRONG_ALERT": 4,
}

# Momentum encoding
MOMENTUM_MAP = {
    "STRONG_DOWN": -2,
    "DOWN": -1,
    "SIDEWAYS": 0,
    "UP": 1,
    "STRONG_UP": 2,
}

# Stock type encoding
STOCK_TYPE_MAP = {
    "SUPER_LOW_FLOAT": 0,
    "LOW_FLOAT": 1,
    "MID_FLOAT": 2,
    "HIGH_FLOAT": 3,
}

# Features extracted from each alert row
WATCHLIST_FEATURE_COLUMNS = [
    # Core price/volume
    "price",
    "volume_log",                # log(volume) for scale
    "price_change_pct",          # % change from previous alert for same symbol
    # Alert metadata
    "alert_type_encoded",
    "quality_score",
    # Momentum & velocity
    "momentum_encoded",
    "velocity_log",              # log(velocity + 1)
    # Trade flow features
    "buy_pressure_pct",
    "net_flow_log",              # log(abs(net_flow) + 1) * sign
    "avg_trade_size_log",
    "trade_count_log",
    # Float info
    "stock_type_encoded",
    "float_shares_log",          # log(float_shares)
    # Filter context / relative metrics
    "net_flow_multiple",
    "trade_size_multiple",
    "buy_pressure_multiple",
    # Time features
    "hour_sin",
    "hour_cos",
    "minutes_since_market_open",
    # Derived
    "dollar_volume_log",         # log(price * volume)
    "float_turnover_pct",        # volume / float_shares * 100
]


def safe_float(val, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_log(val: float, min_val: float = 1.0) -> float:
    """Log transform with floor to avoid log(0)."""
    return np.log(max(abs(val), min_val))


def signed_log(val: float) -> float:
    """Log transform preserving sign."""
    return np.sign(val) * np.log(abs(val) + 1)


def extract_alert_features(row: dict) -> Dict[str, float]:
    """Extract numerical features from a single watchlist_alerts row.

    Args:
        row: dict with keys: price, volume, alert_type, timestamp, details (dict)

    Returns:
        Dict of feature_name -> float value
    """
    details = row.get("details") or {}
    ndv = details.get("net_dollar_volume") or {}
    float_info = details.get("float_info") or {}
    filter_ctx = details.get("filter_context") or {}

    price = safe_float(row.get("price"), 1.0)
    volume = safe_float(row.get("volume"), 0)

    # Core
    features = {
        "price": price,
        "volume_log": safe_log(volume),
        "price_change_pct": 0.0,  # filled later in sequence context

        # Alert metadata
        "alert_type_encoded": float(ALERT_TYPE_MAP.get(
            row.get("alert_type", ""), 2)),
        "quality_score": safe_float(details.get("quality_score"), 5) / 11.0,  # normalize to 0-1

        # Momentum & velocity
        "momentum_encoded": float(MOMENTUM_MAP.get(
            details.get("momentum", ""), 0)),
        "velocity_log": signed_log(safe_float(details.get("velocity"), 0)),

        # Trade flow
        "buy_pressure_pct": safe_float(
            ndv.get("buy_pressure_pct"), 50) / 100.0,  # normalize to 0-1
        "net_flow_log": signed_log(safe_float(
            ndv.get("net_total_dollars"), 0)),
        "avg_trade_size_log": safe_log(safe_float(
            details.get("avg_trade_size"), 100)),
        "trade_count_log": safe_log(safe_float(
            details.get("trade_count"), 0)),

        # Float info
        "stock_type_encoded": float(STOCK_TYPE_MAP.get(
            float_info.get("stock_type", ""), 1)),
        "float_shares_log": safe_log(safe_float(
            float_info.get("float_shares"), 1e6)),

        # Filter context
        "net_flow_multiple": min(safe_float(
            filter_ctx.get("net_flow_multiple"), 1.0), 20.0),
        "trade_size_multiple": min(safe_float(
            filter_ctx.get("trade_size_multiple"), 1.0), 20.0),
        "buy_pressure_multiple": min(safe_float(
            filter_ctx.get("buy_pressure_multiple"), 1.0), 5.0),
    }

    # Time features from timestamp
    ts = row.get("timestamp")
    if ts is not None:
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
        hour = ts.hour if hasattr(ts, "hour") else 12
        minute = ts.minute if hasattr(ts, "minute") else 0
        total_minutes = hour * 60 + minute
        market_open_minutes = 9 * 60 + 30  # 9:30 AM
        features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        features["minutes_since_market_open"] = (total_minutes - market_open_minutes) / 390.0  # normalize by trading day length
    else:
        features["hour_sin"] = 0.0
        features["hour_cos"] = 1.0
        features["minutes_since_market_open"] = 0.0

    # Derived
    features["dollar_volume_log"] = safe_log(price * volume)
    float_shares = safe_float(float_info.get("float_shares"), 1e6)
    features["float_turnover_pct"] = min(
        (volume / max(float_shares, 1)) * 100, 500.0)  # cap at 500%

    return features


def build_watchlist_dataframe(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a DataFrame of watchlist_alerts rows into a feature DataFrame.

    Args:
        alerts_df: DataFrame with columns: id, symbol, alert_type, price, volume, details, timestamp

    Returns:
        DataFrame with one row per alert, columns = WATCHLIST_FEATURE_COLUMNS
    """
    records = []
    for _, row in alerts_df.iterrows():
        row_dict = row.to_dict()
        feats = extract_alert_features(row_dict)
        feats["_id"] = row_dict.get("id")
        feats["_symbol"] = row_dict.get("symbol")
        feats["_timestamp"] = row_dict.get("timestamp")
        feats["_price_raw"] = safe_float(row_dict.get("price"), 0)
        records.append(feats)

    df = pd.DataFrame(records)

    # Compute price_change_pct within each symbol group
    df = df.sort_values(["_symbol", "_timestamp"]).reset_index(drop=True)
    df["price_change_pct"] = df.groupby("_symbol")["_price_raw"].pct_change().fillna(0)
    # Clip extreme values
    df["price_change_pct"] = df["price_change_pct"].clip(-1.0, 1.0)

    return df


def build_sequences(
    df: pd.DataFrame,
    seq_len: int = 3,
    forward_alerts: int = 3,
) -> tuple:
    """Build sequences of alerts and labels for training.

    For each group of `seq_len` consecutive alerts for a symbol,
    the label is the price return `forward_alerts` alerts later.

    Args:
        df: Feature DataFrame from build_watchlist_dataframe()
        seq_len: Number of past alerts in each sequence (default: 3)
        forward_alerts: How many alerts ahead to measure return (default: 3)

    Returns:
        X: np.ndarray of shape (N, seq_len, n_features)
        y: np.ndarray of shape (N,) - forward return as fraction
        meta: list of dicts with symbol, timestamp, price for each sample
    """
    feature_cols = WATCHLIST_FEATURE_COLUMNS
    X_list = []
    y_list = []
    meta_list = []

    for symbol, group in df.groupby("_symbol"):
        group = group.sort_values("_timestamp").reset_index(drop=True)
        if len(group) < seq_len + forward_alerts:
            continue

        prices = group["_price_raw"].values
        features = group[feature_cols].values.astype(np.float32)
        timestamps = group["_timestamp"].values

        for i in range(len(group) - seq_len - forward_alerts + 1):
            seq = features[i : i + seq_len]
            current_price = prices[i + seq_len - 1]
            future_price = prices[i + seq_len - 1 + forward_alerts]

            if current_price <= 0:
                continue

            forward_return = (future_price - current_price) / current_price
            # Clip extreme returns (small caps can have huge moves)
            forward_return = np.clip(forward_return, -0.5, 0.5)

            X_list.append(seq)
            y_list.append(forward_return)
            meta_list.append({
                "symbol": symbol,
                "timestamp": timestamps[i + seq_len - 1],
                "price": current_price,
            })

    if not X_list:
        return np.array([]), np.array([]), []

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)

    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, meta_list
