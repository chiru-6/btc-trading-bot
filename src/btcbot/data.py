"""Market data: download BTC candles from Coinbase's public API, or generate synthetic data.

No API key is needed -- Coinbase Exchange exposes historical candles publicly.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

GRANULARITIES = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}
COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
MAX_CANDLES_PER_REQUEST = 300
COLUMNS = ["open", "high", "low", "close", "volume"]


def fetch_coinbase(
    product: str = "BTC-USD",
    granularity: str = "1d",
    days: int = 3650,
    end: datetime | None = None,
    pause: float = 0.35,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Download OHLCV candles in chunks of 300 (the API limit per request)."""
    import requests  # imported lazily so the rest of the package works offline

    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {list(GRANULARITIES)}")
    sec = GRANULARITIES[granularity]
    end = end or datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    chunk = timedelta(seconds=sec * MAX_CANDLES_PER_REQUEST)
    url = COINBASE_URL.format(product=product)

    rows: list[list[float]] = []
    session = requests.Session()
    cur = start
    retries = 0
    while cur < end:
        nxt = min(cur + chunk, end)
        resp = session.get(
            url,
            params={"granularity": sec, "start": cur.isoformat(), "end": nxt.isoformat()},
            headers={"User-Agent": "btc-trading-bot (educational)"},
            timeout=20,
        )
        if resp.status_code == 429 and retries < 5:  # rate limited -> back off and retry
            retries += 1
            time.sleep(2 * retries)
            continue
        resp.raise_for_status()
        retries = 0
        rows.extend(resp.json())
        cur = nxt
        time.sleep(pause)

    if not rows:
        raise RuntimeError("No candles returned -- check the product id and date range.")

    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop_duplicates("time").sort_values("time").set_index("time")[COLUMNS].astype(float)

    if drop_incomplete:  # the newest candle is usually still forming
        now = datetime.now(timezone.utc)
        if len(df) and df.index[-1] + pd.Timedelta(seconds=sec) > now:
            df = df.iloc[:-1]
    return df


def make_synthetic(n: int = 3000, seed: int = 7, freq: str = "1D") -> pd.DataFrame:
    """Regime-switching random walk that looks roughly like BTC (fat tails, bull/bear phases).

    Useful for running the whole pipeline offline and in CI. A model should find
    only a small, regime-driven edge here -- if it finds a huge one, something leaks.
    """
    rng = np.random.default_rng(seed)
    regimes = np.empty(n, dtype=int)
    regimes[0] = 0
    for i in range(1, n):  # sticky Markov chain: 0 = bull, 1 = bear, 2 = chop
        regimes[i] = regimes[i - 1] if rng.random() < 0.985 else rng.integers(0, 3)
    drift = np.array([0.0025, -0.0025, 0.0])[regimes]
    vol = np.array([0.03, 0.04, 0.025])[regimes]
    rets = drift + vol * rng.standard_t(df=4, size=n) / np.sqrt(2)

    close = 10_000 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, vol, n)) * close
    high = np.maximum(open_, close) + spread / 2
    low = np.minimum(open_, close) - spread / 2
    volume = rng.lognormal(10, 0.5, n) * (1 + 5 * np.abs(rets))

    idx = pd.date_range("2016-01-01", periods=n, freq=freq, tz="UTC", name="time")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="time", parse_dates=["time"])
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df[COLUMNS].astype(float).sort_index()


def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Infer the annualisation factor from candle spacing (crypto trades 24/7)."""
    step = pd.Series(index).diff().median().total_seconds()
    return 365 * 86400 / step
