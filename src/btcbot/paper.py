"""Paper trading: run the trained model on new candles with a simulated portfolio.

Nothing here talks to an exchange's order API. The bot only reads public prices
and records what it *would* have done in a local JSON file.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import positions_from_proba
from .features import build_features

HISTORY_FOR_FEATURES = 600  # candles kept when recomputing features (covers all lookbacks)


@dataclass
class PaperState:
    cash: float = 1000.0
    btc: float = 0.0
    position: int = 0
    last_candle: str | None = None
    starting_cash: float = 1000.0
    trades: int = 0
    equity_log: list = field(default_factory=list)

    @classmethod
    def load(cls, path: Path, starting_cash: float = 1000.0) -> "PaperState":
        if path.exists():
            return cls(**json.loads(path.read_text()))
        return cls(cash=starting_cash, starting_cash=starting_cash)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    def equity(self, price: float) -> float:
        return self.cash + self.btc * price


def paper_step(df: pd.DataFrame, model, meta: dict, state: PaperState,
               trade_log: Path | None = None) -> dict:
    """Process the newest *closed* candle in `df`. Safe to call repeatedly (idempotent)."""
    window = df.tail(HISTORY_FOR_FEATURES)
    feats = build_features(window)[meta["features"]]
    t = feats.index[-1]
    row = feats.iloc[[-1]]
    price = float(window["close"].iloc[-1])

    if state.last_candle == str(t):
        return {"time": str(t), "status": "already processed", "equity": state.equity(price)}
    if row.isna().any(axis=1).iloc[0]:
        return {"time": str(t), "status": "not enough history for features"}

    proba = float(model.predict_proba(row)[:, 1][0])
    target = int(positions_from_proba(
        pd.Series([proba], index=[t]), meta["enter"], meta["exit"], start_position=state.position
    ).iloc[0])

    cost = (meta["fee_bps"] + meta["slippage_bps"]) / 1e4
    action = "hold"
    if target == 1 and state.position == 0:
        state.btc = state.cash * (1 - cost) / price
        state.cash = 0.0
        action = "BUY"
    elif target == 0 and state.position == 1:
        state.cash = state.btc * price * (1 - cost)
        state.btc = 0.0
        action = "SELL"
    if action != "hold":
        state.trades += 1
    state.position = target
    state.last_candle = str(t)
    equity = state.equity(price)
    state.equity_log.append([str(t), round(equity, 2)])

    result = {"time": str(t), "price": price, "p_up": round(proba, 4), "action": action,
              "position": target, "equity": round(equity, 2)}
    if trade_log is not None:
        trade_log.parent.mkdir(parents=True, exist_ok=True)
        new = not trade_log.exists()
        with trade_log.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(result))
            if new:
                w.writeheader()
            w.writerow(result)
    return result


def replay(df: pd.DataFrame, model, meta: dict, state: PaperState, last_n: int,
           trade_log: Path | None = None) -> list[dict]:
    """Simulate running the bot live over the last `last_n` candles of a CSV."""
    results = []
    for end in range(len(df) - last_n, len(df)):
        results.append(paper_step(df.iloc[: end + 1], model, meta, state, trade_log))
    return results


def buy_hold_equity(df: pd.DataFrame, last_n: int, starting_cash: float) -> float:
    closes = df["close"].iloc[-last_n:]
    return float(starting_cash * closes.iloc[-1] / closes.iloc[0]) if len(closes) else np.nan
