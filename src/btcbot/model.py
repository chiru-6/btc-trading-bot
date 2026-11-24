"""Models and walk-forward training.

Walk-forward (expanding window) mimics live use: train on the past, predict the
next block, roll forward. An embargo equal to the label horizon removes training
rows whose labels overlap the test period, so nothing from the future leaks in.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_KINDS = ("logreg", "rf", "gbm")


def make_model(kind: str = "gbm", seed: int = 42):
    if kind == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=2000)),
        ])
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=25, n_jobs=-1, random_state=seed
        )
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50,
            l2_regularization=1.0, random_state=seed,
        )
    raise ValueError(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}")


@dataclass
class Fold:
    fold: int
    train_rows: int
    test_start: str
    test_end: str
    test_rows: int
    accuracy: float
    auc: float
    base_rate: float


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    kind: str = "gbm",
    n_splits: int = 5,
    min_train: int | None = None,
    embargo: int = 1,
    seed: int = 42,
):
    """Return out-of-sample probabilities (NaN where never tested) and per-fold stats."""
    n = len(X)
    min_train = min_train or max(n // 3, 100)
    if min_train + n_splits > n:
        raise ValueError(f"not enough rows ({n}) for min_train={min_train} and {n_splits} splits")
    test_size = (n - min_train) // n_splits

    proba = pd.Series(np.nan, index=X.index, name=f"proba_{kind}")
    folds: list[Fold] = []
    for i in range(n_splits):
        t0 = min_train + i * test_size
        t1 = n if i == n_splits - 1 else t0 + test_size
        train_end = t0 - embargo
        model = make_model(kind, seed)
        model.fit(X.iloc[:train_end], y.iloc[:train_end])
        p = model.predict_proba(X.iloc[t0:t1])[:, 1]
        proba.iloc[t0:t1] = p

        y_test = y.iloc[t0:t1]
        auc = roc_auc_score(y_test, p) if y_test.nunique() == 2 else float("nan")
        folds.append(Fold(
            fold=i, train_rows=train_end,
            test_start=str(X.index[t0]), test_end=str(X.index[t1 - 1]), test_rows=t1 - t0,
            accuracy=float(accuracy_score(y_test, p >= 0.5)), auc=float(auc),
            base_rate=float(y_test.mean()),
        ))
    return proba, pd.DataFrame([asdict(f) for f in folds])


def fit_final(X: pd.DataFrame, y: pd.Series, kind: str = "gbm", seed: int = 42):
    model = make_model(kind, seed)
    model.fit(X, y)
    return model


def save_bundle(path: str | Path, model, meta: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, path)
    return path


def load_bundle(path: str | Path):
    bundle = joblib.load(path)
    return bundle["model"], bundle["meta"]
