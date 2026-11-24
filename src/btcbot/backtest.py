"""Long/flat backtester with fees, slippage and standard performance metrics.

Timing convention: the position decided at the close of candle t earns the
return from close t to close t+1. Costs are charged whenever the position changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def positions_from_proba(proba: pd.Series, enter: float = 0.55, exit: float = 0.50,
                         start_position: int = 0) -> pd.Series:
    """Go long when P(up) >= enter, go flat when P(up) < exit.

    The gap between the thresholds (hysteresis) stops the bot flipping in and out
    on noise, which matters a lot once fees are included.
    """
    if exit > enter:
        raise ValueError("exit threshold must be <= enter threshold")
    pos = np.zeros(len(proba))
    cur = start_position
    for i, p in enumerate(proba.to_numpy()):
        if np.isnan(p):
            cur = 0
        elif cur == 0 and p >= enter:
            cur = 1
        elif cur == 1 and p < exit:
            cur = 0
        pos[i] = cur
    return pd.Series(pos, index=proba.index, name="position")


def run_backtest(close: pd.Series, position: pd.Series, fee_bps: float = 10,
                 slippage_bps: float = 5) -> pd.DataFrame:
    """Return a frame with per-period strategy and buy-and-hold returns plus equity curves."""
    close = close.loc[position.index]
    next_ret = close.pct_change().shift(-1)  # return earned by holding from t to t+1
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    cost = turnover * (fee_bps + slippage_bps) / 1e4

    out = pd.DataFrame({
        "position": position,
        "market_ret": next_ret,
        "cost": cost,
    })
    out["strategy_ret"] = out["position"] * out["market_ret"] - out["cost"]
    out = out.dropna(subset=["market_ret"])
    out["equity"] = (1 + out["strategy_ret"]).cumprod()
    out["buy_hold"] = (1 + out["market_ret"]).cumprod()
    return out


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def trade_stats(bt: pd.DataFrame) -> dict:
    """Per-trade win rate: a trade is a contiguous run of position == 1."""
    pos = bt["position"].to_numpy()
    rets = bt["strategy_ret"].to_numpy()
    trades, cur = [], None
    for p, r in zip(pos, rets):
        if p == 1:
            cur = (1 + r) * (cur if cur is not None else 1.0)
        elif cur is not None:
            trades.append(cur - 1)
            cur = None
    if cur is not None:
        trades.append(cur - 1)
    trades = np.array(trades)
    return {
        "n_trades": int(len(trades)),
        "win_rate": float((trades > 0).mean()) if len(trades) else float("nan"),
        "avg_trade": float(trades.mean()) if len(trades) else float("nan"),
    }


def metrics(returns: pd.Series, periods_per_year: float) -> dict:
    r = returns.dropna()
    if r.empty:
        return {}
    equity = (1 + r).cumprod()
    years = len(r) / periods_per_year
    total = float(equity.iloc[-1] - 1)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    vol = float(r.std() * np.sqrt(periods_per_year))
    downside = r[r < 0].std() * np.sqrt(periods_per_year)
    mean_ann = r.mean() * periods_per_year
    return {
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": float(mean_ann / vol) if vol > 0 else float("nan"),
        "sortino": float(mean_ann / downside) if downside and downside > 0 else float("nan"),
        "max_drawdown": _max_drawdown(equity),
    }


def summarize(bt: pd.DataFrame, periods_per_year: float) -> dict:
    strat = metrics(bt["strategy_ret"], periods_per_year)
    strat.update(trade_stats(bt))
    strat["exposure"] = float(bt["position"].mean())
    strat["total_costs"] = float(bt["cost"].sum())
    return {
        "strategy": strat,
        "buy_and_hold": metrics(bt["market_ret"], periods_per_year),
        "start": str(bt.index[0]),
        "end": str(bt.index[-1]),
        "periods": int(len(bt)),
    }
