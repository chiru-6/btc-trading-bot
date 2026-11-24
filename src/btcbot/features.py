"""Feature engineering and labels.

Every feature at row t uses only data available at the close of candle t.
`tests/test_features.py` enforces this by checking that truncating the future
never changes past feature values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WARMUP = 200  # longest lookback used below


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    f = pd.DataFrame(index=df.index)
    logret = np.log(c).diff()

    # Momentum over several horizons
    for k in (1, 3, 7, 14, 30):
        f[f"ret_{k}"] = np.log(c / c.shift(k))

    # Volatility and volatility regime
    f["vol_7"] = logret.rolling(7).std()
    f["vol_30"] = logret.rolling(30).std()
    f["vol_ratio"] = f["vol_7"] / f["vol_30"]

    # Trend: distance from moving averages
    for w in (10, 50, 200):
        f[f"sma_gap_{w}"] = c / c.rolling(w).mean() - 1

    # Oscillators
    f["rsi_14"] = rsi(c, 14) / 100
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    # Candle shape
    rng = (h - l).replace(0, np.nan)
    f["range"] = (h - l) / c
    f["close_pos"] = (c - l) / rng

    # Volume anomaly
    lv = np.log1p(v)
    f["volume_z"] = (lv - lv.rolling(30).mean()) / lv.rolling(30).std()

    # Drawdown from recent high
    f["dd_90"] = c / c.rolling(90).max() - 1

    return f.replace([np.inf, -np.inf], np.nan)


def make_labels(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0):
    """Target = 1 if the log return over the next `horizon` candles beats `threshold`."""
    fwd = np.log(df["close"].shift(-horizon) / df["close"])
    y = (fwd > threshold).astype(float).where(fwd.notna())
    return y, fwd


def make_dataset(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0):
    """Return (X, y) for rows that have both full features and a known label."""
    X = build_features(df)
    y, _ = make_labels(df, horizon, threshold)
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask].astype(int)


FEATURE_NAMES = list(build_features(pd.DataFrame(
    {k: [1.0] for k in ("open", "high", "low", "close", "volume")})).columns)
