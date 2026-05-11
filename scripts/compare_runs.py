"""
Produce a side-by-side comparison of training runs.

Reads `results/metrics.json` (current) and the latest snapshot in
`results/snapshots/` (previous), prints a markdown table to stdout, and
writes `results/RUN_COMPARISON.md`.

Intended for showing v2.0 (imbalanced, v1 controls) vs v2.1 (balanced,
MAF-matched 1000G controls) side-by-side.
"""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def load(p):
    return json.loads(Path(p).read_text())


def fmt(v, d=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def row(label, a, b, d=3):
    return f"| {label} | {fmt(a, d)} | {fmt(b, d)} |"


def build_table(prev, curr, prev_label="v2.0", curr_label="v2.1"):
    pb = prev["binary"]
    cb = curr["binary"]
    pm = prev["multiclass"]
    cm = curr["multiclass"]

    lines = []
    lines.append(f"# Run comparison: {prev_label} vs {curr_label}\n")
    lines.append(f"Snapshot: `results/snapshots/...` | Current: `results/metrics.json`\n")

    lines.append("## Binary task — aging vs control\n")
    lines.append(f"| Metric | {prev_label} | {curr_label} |")
    lines.append("|---|---:|---:|")
    lines.append(row("Train N",                 pb.get("n_train"),             cb.get("n_train"), d=0))
    lines.append(row("Test N",                  pb.get("n_test"),              cb.get("n_test"), d=0))
    lines.append(row("Train positives (aging)", pb["cv"].get("n_positive"),    cb["cv"].get("n_positive"), d=0))
    lines.append(row("Gene groups (train)",     pb.get("n_gene_groups_train"), cb.get("n_gene_groups_train"), d=0))
    lines.append("")
    lines.append(f"| Metric | {prev_label} (test) | {curr_label} (test) |")
    lines.append("|---|---:|---:|")
    lines.append(row("Balanced accuracy", pb["test"]["balanced_accuracy"], cb["test"]["balanced_accuracy"]))
    lines.append(row("ROC-AUC",           pb["test"]["roc_auc"],           cb["test"]["roc_auc"]))
    lines.append(row("PR-AUC",            pb["test"]["pr_auc"],            cb["test"]["pr_auc"]))
    lines.append(row("Brier",             pb["test"]["brier"],             cb["test"]["brier"]))
    lines.append(row("F1 (aging)",        pb["test"]["f1"],                cb["test"]["f1"]))
    lines.append(row("Raw accuracy",      pb["test"]["accuracy"],          cb["test"]["accuracy"]))
    lines.append("")
    lines.append(f"| Metric | {prev_label} | {curr_label} |")
    lines.append("|---|---:|---:|")
    lines.append(row("Permutation p",       pb.get("permutation_p"),     cb.get("permutation_p"), d=4))
    lines.append(row("Naive CV AUC",        pb.get("naive_cv_auc"),      cb.get("naive_cv_auc")))
    lines.append(row("Grouped CV AUC",      pb.get("grouped_cv_auc"),    cb.get("grouped_cv_auc")))
    lines.append(row("Leakage delta AUC",   pb.get("leakage_delta_auc"), cb.get("leakage_delta_auc")))

    lines.append("\n## Multi-class task — trait category\n")
    lines.append(f"| Metric | {prev_label} | {curr_label} |")
    lines.append("|---|---:|---:|")
    lines.append(row("N classes",              pm.get("n_classes"),                  cm.get("n_classes"), d=0))
    lines.append(row("Test accuracy",          pm.get("test_accuracy"),              cm.get("test_accuracy")))
    lines.append(row("Test balanced accuracy", pm.get("test_balanced_accuracy"),     cm.get("test_balanced_accuracy")))
    lines.append(row("Majority baseline acc",  pm.get("majority_baseline_accuracy"), cm.get("majority_baseline_accuracy")))
    lines.append(row("Random baseline acc",    pm.get("random_baseline_accuracy"),   cm.get("random_baseline_accuracy")))

    lines.append("\n## Interpretation\n")
    p_bacc = pb["test"]["balanced_accuracy"]
    c_bacc = cb["test"]["balanced_accuracy"]
    p_auc  = pb["test"]["roc_auc"]
    c_auc  = cb["test"]["roc_auc"]
    lines.append(f"- Balanced accuracy: **{p_bacc:.3f} → {c_bacc:.3f}** ({c_bacc - p_bacc:+.3f})")
    lines.append(f"- ROC-AUC:           **{p_auc:.3f} → {c_auc:.3f}** ({c_auc - p_auc:+.3f})")
    prev_ratio = (pb['cv']['n'] - pb['cv']['n_positive']) / pb['cv']['n_positive']
    curr_ratio = (cb['cv']['n'] - cb['cv']['n_positive']) / cb['cv']['n_positive']
    lines.append(f"- Class ratio (ctrl:aging) train: {prev_ratio:.2f} → {curr_ratio:.2f}")

    return "\n".join(lines) + "\n"


def main():
    curr_path = ROOT / "results" / "metrics.json"
    snap_dir = ROOT / "results" / "snapshots"
    if not curr_path.exists():
        print(f"FATAL: no current metrics at {curr_path}")
        return
    snaps = sorted(snap_dir.glob("metrics_*.json"))
    if not snaps:
        print(f"FATAL: no snapshot in {snap_dir}")
        return
    prev_path = snaps[-1]

    prev = load(prev_path)
    curr = load(curr_path)

    md = build_table(prev, curr,
                     prev_label=prev_path.stem.replace("metrics_", ""),
                     curr_label="current")

    out = ROOT / "results" / "RUN_COMPARISON.md"
    out.write_text(md)
    print(md)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
