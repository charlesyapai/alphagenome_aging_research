"""
Assign each variant to its nearest/overlapping gene for gene-grouped CV.

Rationale:
    Variants in the same gene are not statistically independent. Naive
    random K-fold cross-validation lets variants from the same gene end up
    in both train and test folds, which inflates metrics via:
      - linkage disequilibrium (nearby variants share haplotypes)
      - cis-regulatory effects predicted identically by AlphaGenome
    GroupKFold by gene prevents this.

Source of gene assignment:
    Preferred: VEP annotation `gene_symbol` (already collected in v1)
    Fallback:  aging_variants.csv `nearest_gene` column
    Last resort: bucket by chromosome position (100kb windows) as a
                 pseudo-group so CV still has some locality constraint.

Output:
    data/processed/gene_groups.csv with columns:
        rsid, gene_group, group_source
"""
import argparse
from pathlib import Path

import pandas as pd


def assign_groups(variants_csv: Path, vep_csv: Path, out_csv: Path):
    variants = pd.read_csv(variants_csv)
    vep = pd.read_csv(vep_csv, usecols=["rsid", "gene_symbol"])

    merged = variants.merge(vep, on="rsid", how="left")

    def pick(row):
        # Preferred
        gs = row.get("gene_symbol")
        if pd.notna(gs) and str(gs).strip():
            return str(gs).strip(), "vep"
        # Fallback
        ng = row.get("nearest_gene")
        if pd.notna(ng) and str(ng).strip():
            return str(ng).strip(), "nearest_gene"
        # Last resort: 100kb bucket
        chrom = row.get("chromosome", "chr?")
        pos = row.get("position", 0)
        try:
            bucket = int(pos) // 100_000
        except (TypeError, ValueError):
            bucket = 0
        return f"{chrom}_bucket_{bucket}", "position_bucket"

    out_rows = []
    for _, r in merged.iterrows():
        group, source = pick(r)
        out_rows.append({
            "rsid": r["rsid"],
            "gene_group": group,
            "group_source": source,
        })

    out = pd.DataFrame(out_rows)
    out.to_csv(out_csv, index=False)

    print(f"Assigned groups for {len(out)} variants")
    print("  By source:")
    print(out["group_source"].value_counts().to_string())
    print(f"  Unique groups: {out['gene_group'].nunique()}")
    print(f"  Saved: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--vep", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assign_groups(Path(args.variants), Path(args.vep), Path(args.out))


if __name__ == "__main__":
    main()
