import numpy as np
import pandas as pd
import pytest

from btcbot.backtest import positions_from_proba, run_backtest, summarize


@pytest.fixture
def close():
    idx = pd.date_range("2024-01-01", periods=6, freq="1D", tz="UTC")
    return pd.Series([100, 110, 99, 120, 120, 132], index=idx, dtype=float)


def test_always_long_without_costs_equals_buy_and_hold(close):
    pos = pd.Series(1.0, index=close.index)
    bt = run_backtest(close, pos, fee_bps=0, slippage_bps=0)
    assert np.isclose(bt["equity"].iloc[-1], 132 / 100)
    assert np.allclose(bt["equity"], bt["buy_hold"])


def test_flat_earns_nothing(close):
    bt = run_backtest(close, pd.Series(0.0, index=close.index))
    assert np.isclose(bt["equity"].iloc[-1], 1.0)


def test_costs_charged_on_each_change(close):
    pos = pd.Series([1, 0, 1, 0, 1, 1], index=close.index, dtype=float)
    free = run_backtest(close, pos, 0, 0)
    paid = run_backtest(close, pos, fee_bps=10, slippage_bps=0)
    assert paid["cost"].sum() == pytest.approx(5 * 0.001)  # 5 changes, last row dropped
    assert paid["equity"].iloc[-1] < free["equity"].iloc[-1]


def test_position_earns_next_period_return(close):
    pos = pd.Series([0, 0, 1, 0, 0, 0], index=close.index, dtype=float)
    bt = run_backtest(close, pos, 0, 0)
    # long at the close of day 2 (99) -> earns the move to day 3 (120)
    assert np.isclose(bt["equity"].iloc[-1], 120 / 99)


def test_hysteresis():
    p = pd.Series([0.50, 0.56, 0.53, 0.51, 0.49, 0.54, 0.60])
    assert positions_from_proba(p, enter=0.55, exit=0.50).tolist() == [0, 1, 1, 1, 0, 0, 1]


def test_summary_has_core_metrics(close):
    bt = run_backtest(close, pd.Series(1.0, index=close.index))
    s = summarize(bt, periods_per_year=365)
    for key in ("sharpe", "max_drawdown", "cagr", "n_trades", "win_rate"):
        assert key in s["strategy"]
    assert s["strategy"]["max_drawdown"] < 0
