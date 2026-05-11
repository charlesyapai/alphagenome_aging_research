"""
Regression tests that assert the pipeline has no leakage.

These codify the v1 → v2 methodology improvements so they can't silently regress.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.pipeline import make_rf_pipeline, make_logreg_pipeline


def test_pipeline_has_imputer_first():
    """Imputer must come before scaler so StandardScaler never sees NaNs."""
    pipe = make_rf_pipeline()
    steps = [name for name, _ in pipe.steps]
    assert steps.index("impute") < steps.index("scale"), \
        f"Imputer must come before scaler, got {steps}"


def test_pipeline_has_scaler_inside():
    """Scaler must be inside the Pipeline object so CV fits it per-fold."""
    pipe = make_rf_pipeline()
    steps = [name for name, _ in pipe.steps]
    assert "scale" in steps, "Pipeline must contain a scaler step"
    assert "impute" in steps, "Pipeline must contain an imputer step"


def test_logreg_pipeline_same_structure():
    pipe = make_logreg_pipeline()
    steps = [name for name, _ in pipe.steps]
    assert steps[:2] == ["impute", "scale"], \
        f"Logreg pipeline must have impute→scale prefix, got {steps}"


def test_pipeline_handles_nans():
    """A Pipeline should fit on NaN-containing input without crashing."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    X[np.isfinite(X)] = X[np.isfinite(X)]  # identity
    X[::7, 0] = np.nan
    y = np.array(["aging"] * 25 + ["control"] * 25)
    pipe = make_rf_pipeline(n_estimators=10)
    pipe.fit(X, y)  # should not raise
    preds = pipe.predict(X)
    assert len(preds) == 50


def test_groupkfold_no_group_leakage():
    """GroupKFold must never put the same group in both train and test."""
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = np.array(["aging"] * 50 + ["control"] * 50)
    groups = np.repeat(np.arange(25), 4)  # 25 groups of 4 samples each

    cv = GroupKFold(n_splits=5)
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        train_groups = set(groups[train_idx])
        test_groups  = set(groups[test_idx])
        assert not (train_groups & test_groups), \
            "Group appears in both train and test — GroupKFold is broken"


def test_groupshufflesplit_no_group_leakage():
    """GroupShuffleSplit (used for held-out test) must also be leakage-free."""
    from sklearn.model_selection import GroupShuffleSplit

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = np.array(["aging"] * 100 + ["control"] * 100)
    groups = np.array([f"gene_{i//8}" for i in range(200)])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    assert not (set(groups[train_idx]) & set(groups[test_idx])), \
        "Gene group leaked across train/test boundary"
