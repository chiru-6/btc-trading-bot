import numpy as np
import pandas as pd

from btcbot.data import make_synthetic
from btcbot.features import FEATURE_NAMES, build_features, make_dataset, make_labels


def test_no_lookahead():
    """Features at time t must not change when data after t is removed or altered."""
    df = make_synthetic(800, seed=1)
    full = build_features(df)
    for k in (300, 450, 799):
        truncated = build_features(df.iloc[: k + 1])
        pd.testing.assert_series_equal(truncated.iloc[-1], full.iloc[k], check_names=False)

    shocked = df.copy()
    shocked.iloc[600:, :] *= 3  # wreck the future
    pd.testing.assert_frame_equal(build_features(shocked).iloc[:600], full.iloc[:600])


def test_labels_look_forward_exactly_horizon():
    df = make_synthetic(100, seed=2)
    y, fwd = make_labels(df, horizon=3)
    expected = np.log(df["close"].iloc[13] / df["close"].iloc[10])
    assert np.isclose(fwd.iloc[10], expected)
    assert y.iloc[-3:].isna().all()  # no label where the future is unknown


def test_dataset_is_clean():
    X, y = make_dataset(make_synthetic(600, seed=3))
    assert list(X.columns) == FEATURE_NAMES
    assert not X.isna().any().any()
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y)
