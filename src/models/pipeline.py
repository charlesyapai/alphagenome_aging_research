"""
sklearn Pipeline factory.

All preprocessing (impute, scale) lives INSIDE the Pipeline so that when the
Pipeline is used inside cross-validation or a train/test split, scaling and
imputation statistics are fit only on training data. This prevents the
leakage that occurred in v1, where StandardScaler was fit on combined
aging+control data before CV.
"""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def make_rf_pipeline(
    n_estimators: int = 300,
    max_depth: int = 10,
    class_weight: str = "balanced",
    random_state: int = 42,
    n_jobs: int = 1,
) -> Pipeline:
    """n_jobs defaults to 1 so the caller (CV, permutation_test_score) can
    parallelize at the fold level without oversubscribing cores."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scale",  StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )),
    ])


def make_logreg_pipeline(
    C: float = 1.0,
    class_weight: str = "balanced",
    random_state: int = 42,
    max_iter: int = 2000,
) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scale",  StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            class_weight=class_weight,
            random_state=random_state,
            max_iter=max_iter,
            n_jobs=-1,
        )),
    ])
