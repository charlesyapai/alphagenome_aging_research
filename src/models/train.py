"""
Train and evaluate the aging-vs-control classifier with honest methodology.

What this does:
    1. Load aging + control feature matrices
    2. Merge with gene groups (for GroupKFold)
    3. Held-out 20% test split BY GENE (not by variant)
    4. 5-fold GroupKFold CV on training set
    5. Report: balanced accuracy, ROC-AUC, PR-AUC, Brier (calibration)
    6. Permutation test (1000 shuffles, within CV scheme)
    7. Fit final model on full train, evaluate on held-out test
    8. Save pipeline artifact + metrics + figures

Why this is different from v1:
    - Preprocessing inside the Pipeline → no leakage across folds
    - GroupKFold by gene → no LD/co-located variant leakage
    - Held-out test set → metrics are on truly unseen data
    - Balanced design (1:1 aging:control by construction) → accuracy is
      informative, not a trivial majority-class prediction
    - Balanced accuracy + AUC + PR-AUC reported, not raw accuracy alone
    - Brier score reported so calibration is visible
"""
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, brier_score_loss, confusion_matrix,
    classification_report, f1_score,
)
from sklearn.model_selection import (
    GroupKFold, GroupShuffleSplit, cross_val_predict, permutation_test_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.pipeline import make_rf_pipeline, make_logreg_pipeline


def load_data(aging_fm_path, control_fm_path, groups_path,
              controls_csv_path=None):
    aging = pd.read_parquet(aging_fm_path)
    control = pd.read_parquet(control_fm_path)
    groups = pd.read_csv(groups_path).set_index("rsid")

    # label column already set in build_features
    if "label" not in aging.columns:
        aging["label"] = "aging"
    if "label" not in control.columns:
        control["label"] = "control"

    df = pd.concat([aging, control], axis=0)

    # Attach gene group. For aging variants it's a direct lookup by rsid.
    df = df.join(groups[["gene_group"]], how="left")

    # For controls from v2.1: each control was matched to a specific aging
    # variant (controls.csv has the `matched_to` column). If we know that
    # match, assign the control to the same gene group as its aging
    # counterpart — prevents a control that sits in the same gene as its
    # aging match from leaking across a GroupKFold fold boundary.
    if controls_csv_path is not None and Path(controls_csv_path).exists():
        ctrl_meta = pd.read_csv(controls_csv_path)
        if {"rsid", "matched_to"}.issubset(ctrl_meta.columns):
            ctrl_map = ctrl_meta.set_index("rsid")["matched_to"].to_dict()
            mask = df["gene_group"].isna()
            for rsid in df.index[mask]:
                aging_rsid = ctrl_map.get(rsid)
                if aging_rsid is not None and aging_rsid in groups.index:
                    df.at[rsid, "gene_group"] = groups.at[aging_rsid, "gene_group"]

    # Final fallback: use the rsid as its own group (isolates any remaining
    # orphans into unique groups — no cross-contamination risk).
    df["gene_group"] = df["gene_group"].fillna(df.index.to_series())

    feature_cols = [c for c in df.columns
                    if c not in ("label", "gene_group") and "__" in c]

    X = df[feature_cols].values
    y = df["label"].values
    g = df["gene_group"].values

    return X, y, g, feature_cols, df.index.values


def split_by_gene(X, y, g, test_frac=0.2, random_state=42):
    """80/20 train/test split where no gene group crosses the boundary."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac,
                                 random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=g))
    return train_idx, test_idx


def eval_metrics(y_true, y_pred, y_prob_pos):
    pos_label = "aging"
    y_bin = (y_true == pos_label).astype(int)
    y_pred_bin = (y_pred == pos_label).astype(int)
    return {
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy":  float(balanced_accuracy_score(y_true, y_pred)),
        "f1":                 float(f1_score(y_bin, y_pred_bin)),
        "roc_auc":            float(roc_auc_score(y_bin, y_prob_pos)),
        "pr_auc":             float(average_precision_score(y_bin, y_prob_pos)),
        "brier":              float(brier_score_loss(y_bin, y_prob_pos)),
        "n":                  int(len(y_true)),
        "n_positive":         int(y_bin.sum()),
    }


def plot_confusion(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=["aging", "control"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["aging", "control"])
    ax.set_yticklabels(["aging", "control"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (held-out test)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration(y_true, y_prob_pos, out_path, n_bins=10):
    y_bin = (y_true == "aging").astype(int)
    bins = np.linspace(0, 1, n_bins + 1)
    mids, mean_pred, frac_pos = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob_pos >= lo) & (y_prob_pos < hi)
        if mask.sum() >= 5:
            mids.append((lo + hi) / 2)
            mean_pred.append(y_prob_pos[mask].mean())
            frac_pos.append(y_bin[mask].mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    if mids:
        ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.set_xlabel("Mean predicted P(aging)")
    ax.set_ylabel("Fraction truly aging")
    ax.set_title("Calibration plot (test set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_importances(pipeline, feature_cols, out_path, top_k=25):
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top = imp.head(top_k)
    fig, ax = plt.subplots(figsize=(9, 8))
    top[::-1].plot(kind="barh", ax=ax, color="#6c5ce7")
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top {top_k} features (binary aging vs control)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    imp.to_csv(out_path.with_suffix(".csv"), header=["importance"])


def train_binary(X, y, g, feature_cols, results_dir, models_dir,
                 n_permutations=1000, n_splits=5, random_state=42):
    print("=" * 70)
    print("BINARY TASK: aging vs control")
    print("=" * 70)

    # Split: 80/20 by gene group
    train_idx, test_idx = split_by_gene(X, y, g, test_frac=0.2,
                                        random_state=random_state)
    X_tr, y_tr, g_tr = X[train_idx], y[train_idx], g[train_idx]
    X_te, y_te       = X[test_idx],  y[test_idx]
    print(f"Train: {len(X_tr)}  ({(y_tr=='aging').sum()} aging / "
          f"{(y_tr=='control').sum()} control, {len(set(g_tr))} gene groups)")
    print(f"Test:  {len(X_te)}  ({(y_te=='aging').sum()} aging / "
          f"{(y_te=='control').sum()} control)")
    # Guard: no gene group appears in both
    overlap = set(g[train_idx]) & set(g[test_idx])
    assert not overlap, f"Leak: {len(overlap)} groups in both train and test"

    cv = GroupKFold(n_splits=n_splits)
    # Use smaller forest during CV/permutation for speed — final model uses full
    pipe = make_rf_pipeline(random_state=random_state, n_estimators=100)

    # CV predictions on training set (group-aware)
    print("\n[CV] group-aware 5-fold on training set...")
    y_cv_pred = cross_val_predict(pipe, X_tr, y_tr, groups=g_tr, cv=cv, n_jobs=-1)
    y_cv_prob = cross_val_predict(pipe, X_tr, y_tr, groups=g_tr, cv=cv,
                                  method="predict_proba", n_jobs=-1)
    classes = pipe.fit(X_tr, y_tr).classes_.tolist()  # for index lookup
    aging_col = classes.index("aging")
    cv_metrics = eval_metrics(y_tr, y_cv_pred, y_cv_prob[:, aging_col])
    print(f"  CV balanced_acc={cv_metrics['balanced_accuracy']:.3f}  "
          f"AUC={cv_metrics['roc_auc']:.3f}  PR-AUC={cv_metrics['pr_auc']:.3f}  "
          f"Brier={cv_metrics['brier']:.3f}")

    # Permutation test — manual, because sklearn's permutation_test_score
    # shuffles LABELS WITHIN GROUPS when `groups=` is set. Our groups are
    # mostly size 1, so shuffling within them is a no-op (p=1.0, std=0).
    # We want: shuffle labels globally, keep groups for CV splits.
    print(f"\n[Permutation test] {n_permutations} shuffles (global shuffle, "
          f"grouped CV)...")
    rng = np.random.default_rng(random_state)
    perm_scores = []
    for i in range(n_permutations):
        if (i + 1) % max(1, n_permutations // 5) == 0:
            print(f"  {i+1}/{n_permutations}...", flush=True)
        y_shuf = rng.permutation(y_tr)
        probs = cross_val_predict(pipe, X_tr, y_shuf, groups=g_tr, cv=cv,
                                  method="predict_proba", n_jobs=-1)
        perm_scores.append(float(roc_auc_score(
            (y_shuf == "aging").astype(int),
            probs[:, pipe.fit(X_tr, y_tr).classes_.tolist().index("aging")])))
    perm_scores = np.array(perm_scores)
    obs = float(cv_metrics["roc_auc"])
    p_value = float((np.sum(perm_scores >= obs) + 1) / (len(perm_scores) + 1))
    score = obs
    print(f"  Observed AUC: {obs:.3f}")
    print(f"  Permutation p: {p_value:.4g}  "
          f"(perm mean AUC={perm_scores.mean():.3f}, "
          f"std={perm_scores.std():.3f})")

    # Also compute a naive (non-grouped) CV AUC so we can demonstrate the gap
    from sklearn.model_selection import StratifiedKFold
    naive_cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)
    y_naive_prob = cross_val_predict(make_rf_pipeline(random_state=random_state),
                                     X_tr, y_tr, cv=naive_cv,
                                     method="predict_proba", n_jobs=-1)
    naive_auc = float(roc_auc_score((y_tr == "aging").astype(int),
                                    y_naive_prob[:, aging_col]))
    print(f"\n[Leakage check] naive (non-grouped) CV AUC: {naive_auc:.3f}")
    print(f"                grouped CV AUC:              {cv_metrics['roc_auc']:.3f}")
    print(f"                delta (leakage if positive): "
          f"{naive_auc - cv_metrics['roc_auc']:+.3f}")

    # Fit final on all training data, evaluate on test (full 300 trees)
    print("\n[Final] fitting on full train, evaluating on held-out test...")
    pipe_final = make_rf_pipeline(random_state=random_state, n_estimators=300)
    pipe_final.fit(X_tr, y_tr)

    y_te_pred = pipe_final.predict(X_te)
    y_te_prob = pipe_final.predict_proba(X_te)
    test_metrics = eval_metrics(y_te, y_te_pred,
                                y_te_prob[:, pipe_final.classes_.tolist().index("aging")])
    print(f"  Test balanced_acc={test_metrics['balanced_accuracy']:.3f}  "
          f"AUC={test_metrics['roc_auc']:.3f}  PR-AUC={test_metrics['pr_auc']:.3f}  "
          f"Brier={test_metrics['brier']:.3f}")
    print(classification_report(y_te, y_te_pred))

    # Artifacts
    results_dir = Path(results_dir)
    models_dir = Path(models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "pipeline": pipe_final,
        "feature_cols": feature_cols,
        "classes": pipe_final.classes_.tolist(),
        "trained_on_n": int(len(X_tr)),
    }, models_dir / "binary_rf.joblib")

    plot_confusion(y_te, y_te_pred, results_dir / "figures" / "confusion_binary.png")
    plot_calibration(y_te,
                     y_te_prob[:, pipe_final.classes_.tolist().index("aging")],
                     results_dir / "figures" / "calibration_binary.png")
    plot_importances(pipe_final, feature_cols,
                     results_dir / "figures" / "importance_binary.png")

    return {
        "cv": cv_metrics,
        "test": test_metrics,
        "permutation_p": float(p_value),
        "observed_auc_perm_context": float(score),
        "naive_cv_auc": naive_auc,
        "grouped_cv_auc": cv_metrics["roc_auc"],
        "leakage_delta_auc": float(naive_auc - cv_metrics["roc_auc"]),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_gene_groups_train": int(len(set(g_tr))),
    }


def train_multiclass(X, y_trait, g, feature_cols, results_dir, models_dir,
                     n_splits=5, random_state=42, min_class_size=30):
    print("\n" + "=" * 70)
    print("MULTI-CLASS TASK: trait category (aging-only)")
    print("=" * 70)

    # Filter to classes with enough samples
    counts = pd.Series(y_trait).value_counts()
    keep = counts[counts >= min_class_size].index.tolist()
    mask = pd.Series(y_trait).isin(keep).values
    X, y_trait, g = X[mask], y_trait[mask], g[mask]
    print(f"Classes kept (≥{min_class_size} samples each): {keep}")
    print(f"Samples: {len(X)}")

    train_idx, test_idx = split_by_gene(X, y_trait, g, test_frac=0.2,
                                        random_state=random_state)
    X_tr, y_tr, g_tr = X[train_idx], y_trait[train_idx], g[train_idx]
    X_te, y_te       = X[test_idx],  y_trait[test_idx]

    pipe = make_rf_pipeline(random_state=random_state, n_estimators=400)

    cv = GroupKFold(n_splits=n_splits)
    print("\n[CV] grouped on training set...")
    y_cv_pred = cross_val_predict(pipe, X_tr, y_tr, groups=g_tr, cv=cv, n_jobs=-1)
    cv_acc = float(accuracy_score(y_tr, y_cv_pred))
    cv_bal = float(balanced_accuracy_score(y_tr, y_cv_pred))
    print(f"  CV acc={cv_acc:.3f}  balanced_acc={cv_bal:.3f}")

    print("\n[Final] fit on full train, evaluate on held-out test...")
    pipe_final = make_rf_pipeline(random_state=random_state, n_estimators=400)
    pipe_final.fit(X_tr, y_tr)
    y_te_pred = pipe_final.predict(X_te)
    test_acc = float(accuracy_score(y_te, y_te_pred))
    test_bal = float(balanced_accuracy_score(y_te, y_te_pred))
    print(f"  Test acc={test_acc:.3f}  balanced_acc={test_bal:.3f}")
    print(classification_report(y_te, y_te_pred))

    # Baseline: majority-class prediction
    baseline_acc = float((pd.Series(y_te).value_counts(normalize=True).iloc[0]))
    n_classes = len(keep)
    print(f"  Majority-class baseline acc: {baseline_acc:.3f}")
    print(f"  Random baseline acc (1/K):   {1/n_classes:.3f}")

    results_dir = Path(results_dir)
    models_dir = Path(models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "pipeline": pipe_final,
        "feature_cols": feature_cols,
        "classes": pipe_final.classes_.tolist(),
        "trained_on_n": int(len(X_tr)),
    }, models_dir / "multiclass_rf.joblib")

    # Confusion matrix heatmap
    cm = confusion_matrix(y_te, y_te_pred, labels=keep)
    fig, ax = plt.subplots(figsize=(max(6, len(keep)), max(5, len(keep)*0.8)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(keep))); ax.set_yticks(range(len(keep)))
    ax.set_xticklabels(keep, rotation=45, ha="right")
    ax.set_yticklabels(keep)
    for i in range(len(keep)):
        for j in range(len(keep)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Multi-class trait (test)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(results_dir / "figures" / "confusion_multiclass.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "classes": keep,
        "n_classes": n_classes,
        "cv_accuracy": cv_acc,
        "cv_balanced_accuracy": cv_bal,
        "test_accuracy": test_acc,
        "test_balanced_accuracy": test_bal,
        "majority_baseline_accuracy": baseline_acc,
        "random_baseline_accuracy": float(1 / n_classes),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aging-fm",    default="data/processed/features_aging.parquet")
    ap.add_argument("--control-fm",  default="data/processed/features_controls.parquet")
    ap.add_argument("--groups",      default="data/processed/gene_groups.csv")
    ap.add_argument("--variants",    default="data/processed/aging_variants.csv")
    ap.add_argument("--controls-csv", default="data/processed/controls.csv",
                    help="control metadata with matched_to column for group inheritance")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--models-dir",  default="models")
    ap.add_argument("--n-permutations", type=int, default=1000)
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    (Path(args.results_dir) / "figures").mkdir(parents=True, exist_ok=True)

    # Binary
    X, y, g, feature_cols, rsids = load_data(
        args.aging_fm, args.control_fm, args.groups,
        controls_csv_path=args.controls_csv,
    )
    binary_results = train_binary(
        X, y, g, feature_cols, args.results_dir, args.models_dir,
        n_permutations=args.n_permutations, random_state=args.seed,
    )

    # Multi-class (aging-only, trait category)
    variants = pd.read_csv(args.variants).set_index("rsid")
    aging_mask = np.array([r in variants.index for r in rsids]) & (y == "aging")
    X_ag = X[aging_mask]
    rsid_ag = rsids[aging_mask]
    g_ag = g[aging_mask]
    y_trait = variants.loc[rsid_ag, "trait_category"].fillna("unknown").values

    mc_results = train_multiclass(
        X_ag, y_trait, g_ag, feature_cols, args.results_dir, args.models_dir,
        random_state=args.seed,
    )

    # Save full metrics
    all_metrics = {
        "binary": binary_results,
        "multiclass": mc_results,
        "config": {
            "seed": args.seed,
            "n_permutations": args.n_permutations,
        },
    }
    (Path(args.results_dir) / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2, default=str)
    )
    print(f"\nSaved: {args.results_dir}/metrics.json")
    print(f"       {args.models_dir}/binary_rf.joblib")
    print(f"       {args.models_dir}/multiclass_rf.joblib")


if __name__ == "__main__":
    main()
