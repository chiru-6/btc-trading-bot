"""Command-line entry point: `btcbot fetch | synth | train | paper`."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import data as D
from .backtest import positions_from_proba, run_backtest, summarize
from .features import make_dataset
from .model import MODEL_KINDS, fit_final, load_bundle, save_bundle, walk_forward
from .paper import PaperState, buy_hold_equity, paper_step, replay
from .report import comparison_table, live_report, plot_equity, write_json


def cmd_fetch(a):
    df = D.fetch_coinbase(a.product, a.granularity, a.days)
    path = D.save_csv(df, a.out)
    print(f"Saved {len(df):,} candles ({df.index[0]} -> {df.index[-1]}) to {path}")


def cmd_synth(a):
    df = D.make_synthetic(a.rows, a.seed)
    path = D.save_csv(df, a.out)
    print(f"Saved {len(df):,} synthetic candles to {path}")


def cmd_train(a):
    df = D.load_csv(a.data)
    if a.holdout:
        # Keep the newest candles completely unseen so `paper --replay` is a true test.
        df = df.iloc[: -a.holdout]
        print(f"Holding out the last {a.holdout} candles (not used for training or backtest).")
    ppy = D.periods_per_year(df.index)
    X, y = make_dataset(df, a.horizon, a.threshold)
    print(f"Dataset: {len(X):,} rows x {X.shape[1]} features | up-rate {y.mean():.1%} "
          f"| {X.index[0].date()} -> {X.index[-1].date()}")

    run_dir = Path(a.reports) / (a.name or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    kinds = MODEL_KINDS if a.model == "all" else (a.model,)
    results, curves = {}, {}
    for kind in kinds:
        proba, folds = walk_forward(X, y, kind, a.splits, a.min_train, embargo=a.horizon, seed=a.seed)
        oos = proba.dropna()
        pos = positions_from_proba(oos, a.enter, a.exit)
        bt = run_backtest(df["close"], pos, a.fee_bps, a.slippage_bps)
        summary = summarize(bt, ppy)
        summary["mean_auc"] = float(folds["auc"].mean())
        summary["folds"] = folds.to_dict("records")
        results[kind], curves[kind] = summary, bt
        run_dir.mkdir(parents=True, exist_ok=True)
        folds.to_csv(run_dir / f"folds_{kind}.csv", index=False)
        bt[["equity", "buy_hold", "position"]].to_csv(run_dir / f"curve_{kind}.csv")
        plot_equity(bt, f"{kind.upper()} walk-forward backtest ({Path(a.data).name})",
                    run_dir / f"equity_{kind}.png")

    table = comparison_table(results)
    table.to_csv(run_dir / "comparison.csv")
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print("\nOut-of-sample results (after fees + slippage):")
    print(table.to_string())

    best = max(results, key=lambda k: results[k]["strategy"].get("sharpe") or float("-inf"))
    if len(kinds) > 1:
        print(f"\nBest by Sharpe: {best}. Note: picking the best of several models on the "
              "same test data flatters its score -- confirm with paper trading.")

    meta = {
        "kind": best, "features": list(X.columns), "horizon": a.horizon,
        "threshold": a.threshold, "enter": a.enter, "exit": a.exit,
        "fee_bps": a.fee_bps, "slippage_bps": a.slippage_bps,
        "trained_on": str(a.data), "trained_rows": len(X),
        "trained_until": str(X.index[-1]), "created": datetime.now(timezone.utc).isoformat(),
        "oos_summary": {k: v for k, v in results[best].items() if k != "folds"},
    }
    final = fit_final(X, y, best, a.seed)
    path = save_bundle(a.model_out, final, meta)
    write_json({"config": vars(a), "results": results}, run_dir / "metrics.json")
    print(f"\nSaved final {best} model (trained on all data) to {path}")
    print(f"Reports in {run_dir}/")


def cmd_paper(a):
    model, meta = load_bundle(a.model)
    state_path, log_path = Path(a.state), Path(a.log)
    if a.data:
        df = D.load_csv(a.data)
    else:
        df = D.fetch_coinbase(a.product, a.granularity, days=a.days)

    if a.replay:
        state = PaperState(cash=a.cash, starting_cash=a.cash)  # fresh portfolio for a replay
        log_path.unlink(missing_ok=True)
        window_start = df.index[-a.replay]
        if pd.Timestamp(meta["trained_until"]) >= window_start:
            print("WARNING: the model was trained on part of this replay window, so the result "
                  "is in-sample and over-optimistic. Retrain with `--holdout "
                  f"{a.replay}` for an honest replay.\n")
        res = replay(df, model, meta, state, a.replay, log_path)
        trades = [r for r in res if r.get("action") in ("BUY", "SELL")]
        for r in trades[-10:]:
            print(f"{r['time']}  {r['action']:4}  @ {r['price']:,.2f}  p_up={r['p_up']:.3f}")
        bh = buy_hold_equity(df, a.replay, a.cash)
        print(f"\nReplayed {a.replay} candles: {state.trades} trades | bot equity "
              f"${state.equity(df['close'].iloc[-1]):,.2f} vs buy & hold ${bh:,.2f} "
              f"(start ${a.cash:,.2f})")
        state.save(state_path)
        return

    state = PaperState.load(state_path, a.cash)
    res = paper_step(df, model, meta, state, log_path)
    state.save(state_path)
    print(json.dumps(res, indent=2))


def cmd_live_report(a):
    state = PaperState.load(Path(a.state))
    if not state.equity_log:
        print("No paper-trading history yet.")
        return
    df = D.load_csv(a.data)
    summary = live_report(state, df["close"], Path(a.log), Path(a.out))
    print(json.dumps(summary, indent=2))


def cmd_site(a):
    from .site import build_site
    out = build_site(
        Path(a.out), data_csv=Path(a.data), state_path=Path(a.state), trades_csv=Path(a.log),
        model_path=Path(a.model), curve_csv=Path(a.curve) if a.curve else None, repo=a.repo,
    )
    print(f"Built website in {out}/ (open index.html via a local server: "
          f"python -m http.server -d {out})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="btcbot", description="ML Bitcoin trading research bot (paper only)")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download candles from Coinbase")
    f.add_argument("--product", default="BTC-USD")
    f.add_argument("--granularity", default="1d", choices=list(D.GRANULARITIES))
    f.add_argument("--days", type=int, default=3650)
    f.add_argument("--out", default="data/btc_1d.csv")
    f.set_defaults(func=cmd_fetch)

    s = sub.add_parser("synth", help="generate synthetic candles for offline testing")
    s.add_argument("--rows", type=int, default=3000)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default="data/synthetic.csv")
    s.set_defaults(func=cmd_synth)

    t = sub.add_parser("train", help="walk-forward train, backtest and save a model")
    t.add_argument("--data", default="data/btc_1d.csv")
    t.add_argument("--model", default="all", choices=("all",) + MODEL_KINDS)
    t.add_argument("--horizon", type=int, default=1, help="candles ahead to predict")
    t.add_argument("--threshold", type=float, default=0.0, help="min log return counted as 'up'")
    t.add_argument("--splits", type=int, default=5)
    t.add_argument("--min-train", type=int, default=None, help="rows in first training window")
    t.add_argument("--enter", type=float, default=0.55, help="P(up) needed to buy")
    t.add_argument("--exit", type=float, default=0.50, help="P(up) below which to sell")
    t.add_argument("--fee-bps", type=float, default=10)
    t.add_argument("--slippage-bps", type=float, default=5)
    t.add_argument("--holdout", type=int, default=0,
                   help="exclude the newest N candles entirely (for an honest paper replay)")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--name", default=None, help="report folder name")
    t.add_argument("--reports", default="reports")
    t.add_argument("--model-out", default="models/model.joblib")
    t.set_defaults(func=cmd_train)

    q = sub.add_parser("paper", help="paper-trade the latest candle (or replay history)")
    q.add_argument("--model", default="models/model.joblib")
    q.add_argument("--data", default=None, help="use a CSV instead of fetching live candles")
    q.add_argument("--product", default="BTC-USD")
    q.add_argument("--granularity", default="1d", choices=list(D.GRANULARITIES))
    q.add_argument("--days", type=int, default=400)
    q.add_argument("--replay", type=int, default=0, help="simulate the last N candles")
    q.add_argument("--cash", type=float, default=1000.0)
    q.add_argument("--state", default="paper/state.json")
    q.add_argument("--log", default="paper/trades.csv")
    q.set_defaults(func=cmd_paper)

    r = sub.add_parser("live-report", help="chart + summary of the paper-trading account")
    r.add_argument("--data", default="data/btc_1d.csv", help="candles CSV for buy & hold")
    r.add_argument("--state", default="paper/state.json")
    r.add_argument("--log", default="paper/trades.csv")
    r.add_argument("--out", default="paper")
    r.set_defaults(func=cmd_live_report)

    w = sub.add_parser("site", help="build the static dashboard website")
    w.add_argument("--data", default="data/btc_1d.csv")
    w.add_argument("--state", default="paper/state.json")
    w.add_argument("--log", default="paper/trades.csv")
    w.add_argument("--model", default="models/model.joblib")
    w.add_argument("--curve", default=None, help="backtest curve CSV (curve_<kind>.csv)")
    w.add_argument("--repo", default=None, help="owner/name for the GitHub link")
    w.add_argument("--out", default="_site")
    w.set_defaults(func=cmd_site)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
