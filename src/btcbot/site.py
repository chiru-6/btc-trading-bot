"""Build the static dashboard website (GitHub Pages) from the bot's files.

Output: <out>/index.html (static template) + <out>/data.json (all numbers).
"""
from __future__ import annotations

import json
import math
import os
import shutil
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

import pandas as pd

from . import data as D
from .paper import PaperState


def _clean(obj):
    """Make values JSON-safe (NaN/inf -> None, numpy -> python)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item"):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def build_payload(data_csv: Path, state_path: Path, trades_csv: Path,
                  model_path: Path | None, curve_csv: Path | None,
                  repo: str | None, price_days: int = 180) -> dict:
    df = D.load_csv(data_csv)
    closes = df["close"]
    payload: dict = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "repo": repo,
        "live": None,
        "model": None,
        "backtest": None,
        "price": [{"t": _day(t), "close": round(float(c), 2)}
                  for t, c in closes.tail(price_days).items()],
    }

    # --- live paper-trading account
    if state_path.exists():
        state = PaperState.load(state_path)
        if state.equity_log:
            log = pd.DataFrame(state.equity_log, columns=["time", "equity"])
            log["time"] = pd.to_datetime(log["time"], utc=True)
            log = log.drop_duplicates("time", keep="last").set_index("time")
            px = closes.reindex(log.index).ffill()
            log["bh"] = state.starting_cash * px / px.iloc[0]
            trades = []
            last_p = None
            if trades_csv.exists():
                t = pd.read_csv(trades_csv)
                if len(t):
                    last_p = float(t["p_up"].iloc[-1])
                    acts = t[t["action"].isin(["BUY", "SELL"])]
                    trades = [{"t": _day(r.time), "action": r.action, "price": float(r.price),
                               "p_up": float(r.p_up), "equity": float(r.equity)}
                              for r in acts.itertuples()]
            last = log.iloc[-1]
            payload["live"] = {
                "since": _day(log.index[0]),
                "last_candle": _day(log.index[-1]),
                "days": int(len(log)),
                "starting_cash": state.starting_cash,
                "bot_equity": float(last["equity"]),
                "bh_equity": float(last["bh"]),
                "position": "LONG" if state.position else "CASH",
                "last_p_up": last_p,
                "n_trades": state.trades,
                "equity": [{"t": _day(t), "bot": round(float(r.equity), 2), "bh": round(float(r.bh), 2)}
                           for t, r in log.iterrows()],
                "trades": trades,
            }

    # --- model card + out-of-sample backtest
    if model_path and model_path.exists():
        from .model import load_bundle
        _, meta = load_bundle(model_path)
        payload["model"] = {k: meta.get(k) for k in (
            "kind", "horizon", "enter", "exit", "fee_bps", "slippage_bps",
            "trained_rows", "trained_until", "created")}
        payload["model"]["n_features"] = len(meta.get("features", []))
        payload["model"]["features"] = meta.get("features", [])
        oos = meta.get("oos_summary") or {}
        if oos:
            payload["backtest"] = {"strategy": oos.get("strategy"),
                                   "buy_and_hold": oos.get("buy_and_hold"),
                                   "start": oos.get("start"), "end": oos.get("end"),
                                   "mean_auc": oos.get("mean_auc"), "curve": []}
    if curve_csv and curve_csv.exists() and payload["backtest"] is not None:
        c = pd.read_csv(curve_csv, index_col=0, parse_dates=True)
        payload["backtest"]["curve"] = [
            {"t": _day(t), "bot": round(float(r.equity), 4), "bh": round(float(r.buy_hold), 4)}
            for t, r in c.iterrows()]

    return _clean(payload)


def build_site(out: Path, **kwargs) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    template = resources.files("btcbot").joinpath("web/index.html")
    with resources.as_file(template) as src:
        shutil.copy(src, out / "index.html")
    repo = kwargs.pop("repo", None) or os.environ.get("GITHUB_REPOSITORY")
    payload = build_payload(repo=repo, **kwargs)
    (out / "data.json").write_text(json.dumps(payload, separators=(",", ":")))
    (out / ".nojekyll").write_text("")
    return out
