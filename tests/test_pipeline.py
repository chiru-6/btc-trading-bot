import numpy as np
import pandas as pd

from btcbot.cli import main
from btcbot.data import make_synthetic, save_csv
from btcbot.features import make_dataset
from btcbot.model import load_bundle, walk_forward


def test_walk_forward_only_predicts_after_min_train():
    X, y = make_dataset(make_synthetic(900, seed=4))
    proba, folds = walk_forward(X, y, "logreg", n_splits=3, min_train=300, embargo=1)
    assert proba.iloc[:300].isna().all()
    assert proba.iloc[300:].notna().all()
    # embargo: training always stops before the test block starts
    for _, f in folds.iterrows():
        assert f["train_rows"] < X.index.get_loc(pd.Timestamp(f["test_start"]))


def test_end_to_end(tmp_path):
    csv = save_csv(make_synthetic(900, seed=5), tmp_path / "synth.csv")
    model_path = tmp_path / "model.joblib"
    main(["train", "--data", str(csv), "--model", "logreg", "--splits", "3",
          "--holdout", "60", "--reports", str(tmp_path / "reports"), "--name", "t",
          "--model-out", str(model_path)])
    assert (tmp_path / "reports" / "t" / "equity_logreg.png").exists()
    model, meta = load_bundle(model_path)
    assert meta["kind"] == "logreg"

    main(["paper", "--model", str(model_path), "--data", str(csv), "--replay", "60",
          "--state", str(tmp_path / "state.json"), "--log", str(tmp_path / "trades.csv")])
    assert (tmp_path / "state.json").exists()


def test_live_report(tmp_path):
    csv = save_csv(make_synthetic(700, seed=6), tmp_path / "synth.csv")
    model_path = tmp_path / "model.joblib"
    main(["train", "--data", str(csv), "--model", "logreg", "--splits", "3", "--holdout", "30",
          "--reports", str(tmp_path / "r"), "--name", "t", "--model-out", str(model_path)])
    state, log = str(tmp_path / "live" / "state.json"), str(tmp_path / "live" / "trades.csv")
    main(["paper", "--model", str(model_path), "--data", str(csv), "--replay", "30",
          "--state", state, "--log", log])
    main(["live-report", "--data", str(csv), "--state", state, "--log", log,
          "--out", str(tmp_path / "live")])
    assert (tmp_path / "live" / "equity.png").exists()
    assert "Live paper-trading status" in (tmp_path / "live" / "README.md").read_text()


def test_site_build(tmp_path):
    import json
    csv = save_csv(make_synthetic(700, seed=8), tmp_path / "synth.csv")
    model_path = tmp_path / "model.joblib"
    main(["train", "--data", str(csv), "--model", "logreg", "--splits", "3", "--holdout", "30",
          "--reports", str(tmp_path / "r"), "--name", "t", "--model-out", str(model_path)])
    state, log = str(tmp_path / "state.json"), str(tmp_path / "trades.csv")
    main(["paper", "--model", str(model_path), "--data", str(csv), "--replay", "30",
          "--state", state, "--log", log])
    main(["site", "--data", str(csv), "--state", state, "--log", log, "--model", str(model_path),
          "--curve", str(tmp_path / "r" / "t" / "curve_logreg.csv"), "--repo", "me/bot",
          "--out", str(tmp_path / "site")])
    d = json.loads((tmp_path / "site" / "data.json").read_text())
    assert (tmp_path / "site" / "index.html").exists()
    assert d["live"]["days"] == 30 and d["backtest"]["curve"] and d["model"]["kind"] == "logreg"
