"""
Aggregate AlphaGenome raw scores into a per-variant feature matrix.

For each variant, AlphaGenome returns ~26,500 rows (5 output types × ~700
biosamples × ~7 genes for RNA-seq). We reduce this to a 113-feature vector:

    For each of 5 output types (RNA_SEQ, ATAC, DNASE, CHIP_HISTONE, CHIP_TF):
        For each of 10 tissue categories:
            - {output}__{tissue}__max       : max |score| across biosamples
            - {output}__{tissue}__mean_abs  : mean |score| across biosamples
        - {output}__global__max
        - {output}__global__mean_abs
        - {output}__global__std

Total: 5 × (10 × 2 + 3) = 115 features per variant (a few global summaries).

Note: this aggregation is intentionally simple — no learned projections, no
PCA on the raw tensor. The features are direct summaries of the pretrained
model's output. The downstream classifier is small (Random Forest / logistic)
so that variance in the pipeline is dominated by AlphaGenome's priors.

Usage:
    python src/features/build_features.py \\
        --scores-dir data/alphagenome_scores/aging \\
        --out data/processed/features_aging.parquet \\
        --label aging
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_TYPES = ["RNA_SEQ", "ATAC", "DNASE", "CHIP_HISTONE", "CHIP_TF"]
TISSUES = [
    "brain", "heart", "liver", "blood", "kidney",
    "muscle", "adipose", "skin", "colon", "lung",
]


def build_variant_features(df: pd.DataFrame) -> dict:
    """Aggregate one variant's long-format scores into a flat feature dict."""
    features = {}

    for ot in OUTPUT_TYPES:
        ot_data = df[df["output_type"] == ot]

        if ot_data.empty:
            for tissue in TISSUES:
                features[f"{ot}__{tissue}__max"] = np.nan
                features[f"{ot}__{tissue}__mean_abs"] = np.nan
            features[f"{ot}__global__max"] = np.nan
            features[f"{ot}__global__mean_abs"] = np.nan
            features[f"{ot}__global__std"] = np.nan
            continue

        all_scores = ot_data["raw_score"].values

        for tissue in TISSUES:
            tissue_data = ot_data[ot_data["tissue_category"] == tissue]
            if tissue_data.empty:
                features[f"{ot}__{tissue}__max"] = np.nan
                features[f"{ot}__{tissue}__mean_abs"] = np.nan
                continue

            scores = tissue_data["raw_score"].values
            abs_scores = np.abs(scores)
            valid_mask = ~np.isnan(abs_scores)
            if valid_mask.any():
                max_idx = np.nanargmax(abs_scores)
                features[f"{ot}__{tissue}__max"] = float(scores[max_idx])
                features[f"{ot}__{tissue}__mean_abs"] = float(np.nanmean(abs_scores))
            else:
                features[f"{ot}__{tissue}__max"] = np.nan
                features[f"{ot}__{tissue}__mean_abs"] = np.nan

        valid_scores = all_scores[~np.isnan(all_scores)]
        if len(valid_scores) > 0:
            abs_all = np.abs(valid_scores)
            max_idx = int(np.argmax(abs_all))
            features[f"{ot}__global__max"] = float(valid_scores[max_idx])
            features[f"{ot}__global__mean_abs"] = float(np.mean(abs_all))
            features[f"{ot}__global__std"] = float(np.std(valid_scores))
        else:
            features[f"{ot}__global__max"] = np.nan
            features[f"{ot}__global__mean_abs"] = np.nan
            features[f"{ot}__global__std"] = np.nan

    return features


def build_feature_matrix(scores_dir: Path, label: str, verbose=True) -> pd.DataFrame:
    score_files = sorted(scores_dir.glob("*.parquet"))
    if verbose:
        print(f"Found {len(score_files)} score files in {scores_dir}")

    all_features = []
    for i, f in enumerate(score_files):
        if verbose and (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(score_files)}] processed")
        try:
            df = pd.read_parquet(f)
            feats = build_variant_features(df)
            feats["rsid"] = f.stem
            feats["label"] = label
            all_features.append(feats)
        except Exception as e:
            print(f"  [error] {f.stem}: {e}")

    fm = pd.DataFrame(all_features).set_index("rsid")
    return fm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True, choices=["aging", "control"])
    args = ap.parse_args()

    fm = build_feature_matrix(Path(args.scores_dir), args.label)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fm.to_parquet(args.out)
    print(f"\nSaved {fm.shape[0]} variants × {fm.shape[1]} features → {args.out}")


if __name__ == "__main__":
    main()
