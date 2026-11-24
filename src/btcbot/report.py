"""Charts and report files for a training run."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.ticker  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_equity(bt: pd.DataFrame, title: str, path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(bt.index, bt["equity"], label="Strategy (out-of-sample, after costs)", lw=1.6)
    ax1.plot(bt.index, bt["buy_hold"], label="Buy & hold", lw=1.2, alpha=0.8)
    ax1.set_yscale("log")
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}")
    ax1.yaxis.set_major_formatter(fmt)
    ax1.yaxis.set_minor_formatter(fmt)
    ax1.set_ylabel("Growth of $1 (log)")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    for col, label in (("equity", "Strategy"), ("buy_hold", "Buy & hold")):
        dd = bt[col] / bt[col].cummax() - 1
        ax2.plot(bt.index, dd * 100, label=label, lw=1)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_json(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, summary in results.items():
        s = summary["strategy"]
        rows.append({
            "model": name,
            "total_return": s.get("total_return"),
            "cagr": s.get("cagr"),
            "sharpe": s.get("sharpe"),
            "max_drawdown": s.get("max_drawdown"),
            "n_trades": s.get("n_trades"),
            "win_rate": s.get("win_rate"),
            "exposure": s.get("exposure"),
            "mean_auc": summary.get("mean_auc"),
        })
    first = next(iter(results.values()))["buy_and_hold"]
    rows.append({"model": "buy_and_hold", "total_return": first["total_return"],
                 "cagr": first["cagr"], "sharpe": first["sharpe"],
                 "max_drawdown": first["max_drawdown"], "exposure": 1.0})
    return pd.DataFrame(rows).set_index("model")


def live_report(state, closes: pd.Series, trades_csv: Path, out_dir: Path) -> dict:
    """Chart + markdown summary of the live paper-trading account vs buy & hold."""
    log = pd.DataFrame(state.equity_log, columns=["time", "equity"])
    log["time"] = pd.to_datetime(log["time"], utc=True)
    log = log.drop_duplicates("time", keep="last").set_index("time")
    px = closes.reindex(log.index).ffill()
    log["buy_hold"] = state.starting_cash * px / px.iloc[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(log.index, log["equity"], label="Bot (paper)", lw=1.8, marker="o" if len(log) < 40 else None)
    ax.plot(log.index, log["buy_hold"], label="Buy & hold BTC", lw=1.2, alpha=0.85)
    ax.axhline(state.starting_cash, color="grey", lw=0.8, ls="--")
    ax.set_ylabel("Account value ($)")
    ax.set_title("Live paper trading since " + str(log.index[0].date()))
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "equity.png", dpi=120)
    plt.close(fig)

    last = log.iloc[-1]
    last_trade = None
    if trades_csv.exists():
        t = pd.read_csv(trades_csv)
        if len(t):
            last_trade = t.iloc[-1].to_dict()
    summary = {
        "since": str(log.index[0].date()),
        "last_update": str(log.index[-1]),
        "days": len(log),
        "bot_equity": round(float(last["equity"]), 2),
        "buy_hold_equity": round(float(last["buy_hold"]), 2),
        "bot_return": float(last["equity"] / state.starting_cash - 1),
        "buy_hold_return": float(last["buy_hold"] / state.starting_cash - 1),
        "trades": state.trades,
        "position": "LONG BTC" if state.position else "CASH",
        "last_p_up": None if last_trade is None else last_trade.get("p_up"),
    }
    md = [
        "# Live paper-trading status",
        "",
        f"_Updated automatically by GitHub Actions. Last candle: {summary['last_update']}_",
        "",
        "| | Bot | Buy & hold |",
        "|---|---:|---:|",
        f"| Account value | ${summary['bot_equity']:,.2f} | ${summary['buy_hold_equity']:,.2f} |",
        f"| Return | {summary['bot_return']:+.2%} | {summary['buy_hold_return']:+.2%} |",
        "",
        f"- Running since **{summary['since']}** ({summary['days']} daily candles)",
        f"- Current position: **{summary['position']}**"
        + (f" (latest P(up) = {summary['last_p_up']:.3f})" if summary["last_p_up"] is not None else ""),
        f"- Trades so far: **{summary['trades']}**",
        "",
        "![live equity](equity.png)",
        "",
        "Simulated $1,000 account, 10 bps fee + 5 bps slippage per trade. No real money.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(md))
    return summary
