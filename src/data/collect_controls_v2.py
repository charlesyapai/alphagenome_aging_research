"""
MAF-matched control selection via remote tabix queries against the
1000 Genomes GRCh38 high-coverage phased panel (2,504 unrelated samples).

Why this replaces collect_controls.py:
    The Ensembl REST approach timed out repeatedly at scale (30-500s per
    variant, 0% hit rate in practice). pysam fetches records directly from
    a remote tabix-indexed VCF using HTTP range requests — the MAF is in
    the INFO/AF field so we don't need a second batch call. In testing:
    3,888 records fetched in 1.6s in a +/-50kb window, 100% with AF.

Data source:
    1kGP_high_coverage_Illumina.chr{N}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz
    hosted at ftp.1000genomes.ebi.ac.uk (assembly GRCh38).

Matching criteria (same as v1 design, but now actually enforced):
    1. Same chromosome (by construction).
    2. MAF bin = MAF(min(AF, 1-AF)) bin of the aging variant, based on
       gnomAD AF from VEP.
    3. Biallelic SNV.
    4. Not in aging rsID set.
    5. Not in the aging positions set (some aging variants have no rsID).
    6. Within +/- WINDOW bp of the aging variant.

Output schema (data/processed/controls.csv) matches aging_variants.csv so
it can drop into the rest of the pipeline:
    rsid, chromosome, position, ref_allele, alt_allele,
    gnomad_af, maf_bin, matched_to, matched_distance_bp, aging_bin

The "rsid" we emit for controls is `chr{N}_{pos}_{ref}_{alt}` since the
high-coverage 2022 panel uses positional IDs, not dbSNP rsIDs.

Resumable via data/processed/controls.csv (skips already-matched aging
rsIDs on restart).
"""
import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
import pysam


VCF_URL_TEMPLATE = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1000G_2504_high_coverage/working/"
    "20220422_3202_phased_SNV_INDEL_SV/"
    "1kGP_high_coverage_Illumina.{chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
)

MAF_BINS = [
    ("<0.1%",   0.0,    0.001),
    ("0.1-1%",  0.001,  0.01),
    ("1-5%",    0.01,   0.05),
    ("5-10%",   0.05,   0.10),
    ("10-50%",  0.10,   0.50 + 1e-9),
]


def to_maf(af):
    if af is None or pd.isna(af):
        return None
    af = float(af)
    return min(af, 1.0 - af)


def maf_bin(m):
    if m is None:
        return None
    for name, lo, hi in MAF_BINS:
        if lo <= m < hi:
            return name
    return None


class VCFCache:
    """Thread-local per-chromosome remote VCF handles.

    pysam VariantFile handles are NOT safe to share across threads even
    with external locking (htslib internal state is per-handle). We keep
    one handle per (thread, chromosome) tuple.
    """
    def __init__(self):
        self._tls = threading.local()

    def get(self, chrom):
        if not chrom.startswith("chr"):
            chrom = "chr" + chrom
        if not hasattr(self._tls, "handles"):
            self._tls.handles = {}
        if chrom not in self._tls.handles:
            url = VCF_URL_TEMPLATE.format(chrom=chrom)
            self._tls.handles[chrom] = pysam.VariantFile(url)
        return self._tls.handles[chrom]


def fetch_candidates(vcf, chrom, pos, window, target_bin,
                     aging_rsids, aging_positions):
    """Fetch candidate biallelic SNVs with matching MAF bin in region."""
    start = max(1, pos - window)
    end = pos + window
    candidates = []
    for rec in vcf.fetch(chrom, start, end):
        # Must be biallelic SNV
        if rec.alts is None or len(rec.alts) != 1:
            continue
        alt = rec.alts[0]
        if len(rec.ref) != 1 or len(alt) != 1:
            continue
        # MAF bin match
        af = rec.info.get("AF")
        if af is None:
            continue
        af = float(af[0]) if isinstance(af, tuple) else float(af)
        m = to_maf(af)
        if maf_bin(m) != target_bin:
            continue
        # Exclude aging variants by rsID and by position
        candidate_rsid = f"{chrom}_{rec.pos}_{rec.ref}_{alt}"
        if rec.id and rec.id in aging_rsids:
            continue
        if (chrom, rec.pos) in aging_positions:
            continue
        candidates.append({
            "rsid": candidate_rsid,
            "chromosome": chrom,
            "position": int(rec.pos),
            "ref_allele": rec.ref,
            "alt_allele": alt,
            "gnomad_af": m,
            "maf_bin": target_bin,
        })
    return candidates


def pick_matched_control(aging_row, vcf_cache, aging_rsids, aging_positions,
                         windows=(50_000, 200_000, 1_000_000)):
    target_maf = to_maf(aging_row.get("gnomad_af"))
    target_bin = maf_bin(target_maf)
    if target_bin is None:
        return None
    chrom = aging_row["chromosome"]
    if not chrom.startswith("chr"):
        chrom = "chr" + chrom
    pos = int(aging_row["position"])

    try:
        vcf = vcf_cache.get(chrom)
    except Exception as e:
        return None

    for w in windows:
        try:
            candidates = fetch_candidates(
                vcf, chrom, pos, w, target_bin, aging_rsids, aging_positions,
            )
        except Exception:
            candidates = []
        if candidates:
            ctrl = random.choice(candidates)
            ctrl["matched_to"] = aging_row["rsid"]
            ctrl["matched_distance_bp"] = abs(ctrl["position"] - pos)
            return ctrl
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aging-csv", default="data/processed/aging_variants.csv")
    ap.add_argument("--vep-csv",   default="data/processed/vep_annotations.csv")
    ap.add_argument("--out",       default="data/processed/controls.csv")
    ap.add_argument("--log",       default="data/processed/control_selection_log.json")
    ap.add_argument("--workers",   type=int, default=8,
                    help="parallel tabix connections (pysam handles are thread-safe "
                         "per handle with a lock; we use one handle per chrom)")
    ap.add_argument("--limit",     type=int, default=0,
                    help="process only the first N aging variants (0 = all)")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--save-every", type=int, default=50)
    args = ap.parse_args()

    random.seed(args.seed)

    aging = pd.read_csv(args.aging_csv)
    vep = pd.read_csv(args.vep_csv, usecols=["rsid", "gnomad_af"])
    aging = aging.merge(vep, on="rsid", how="left")
    aging_rsids = set(aging["rsid"])
    aging_positions = set(
        (r["chromosome"] if str(r["chromosome"]).startswith("chr")
         else "chr" + str(r["chromosome"]),
         int(r["position"]))
        for _, r in aging.iterrows()
    )

    print(f"Loaded {len(aging)} aging variants "
          f"({aging['gnomad_af'].notna().sum()} with gnomad_af)", flush=True)

    aging_match = aging[aging["gnomad_af"].notna()].copy()
    aging_match["target_maf"] = aging_match["gnomad_af"].apply(to_maf)
    aging_match["target_bin"] = aging_match["target_maf"].apply(maf_bin)
    aging_match = aging_match.dropna(subset=["target_bin"])
    print(f"  Matchable: {len(aging_match)}", flush=True)

    if args.limit > 0:
        aging_match = aging_match.head(args.limit)
        print(f"  Limited to first {args.limit}", flush=True)

    out_path = Path(args.out)
    if out_path.exists() and out_path.stat().st_size > 0:
        existing = pd.read_csv(out_path)
        matched_rsids = set(existing["matched_to"])
        results = existing.to_dict("records")
        print(f"Resuming: {len(results)} already matched", flush=True)
    else:
        matched_rsids = set()
        results = []

    # Serialize handle access per-chromosome; parallelize across chromosomes
    vcf_cache = VCFCache()

    to_process = [row for _, row in aging_match.iterrows()
                  if row["rsid"] not in matched_rsids]
    print(f"Processing {len(to_process)} variants with {args.workers} workers",
          flush=True)

    unmatched = []
    t0 = time.time()
    save_lock = Lock()

    def work(row):
        try:
            ctrl = pick_matched_control(
                row, vcf_cache, aging_rsids, aging_positions,
            )
        except Exception as e:
            return row, None, str(e)[:120]
        return row, ctrl, None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, row): row for row in to_process}
        for n, fut in enumerate(as_completed(futures), start=1):
            row, ctrl, err = fut.result()
            if ctrl is not None:
                ctrl["aging_bin"] = row["target_bin"]
                results.append(ctrl)
            else:
                unmatched.append(row["rsid"])

            if n % args.save_every == 0 or n == len(to_process):
                n_done = len(results)
                n_miss = len(unmatched)
                rate = n_done / max(1, (n_done + n_miss))
                elapsed = time.time() - t0
                per = elapsed / n
                eta_min = per * (len(to_process) - n) / 60
                print(f"[{n}/{len(to_process)}] matched={n_done} "
                      f"unmatched={n_miss} hit_rate={rate:.1%} "
                      f"per={per:.2f}s ETA ~{eta_min:.0f}min",
                      flush=True)
                with save_lock:
                    pd.DataFrame(results).to_csv(out_path, index=False)

    pd.DataFrame(results).to_csv(out_path, index=False)

    log = {
        "n_aging_input":         int(len(aging)),
        "n_aging_matchable":     int(len(to_process)),
        "n_controls_matched":    int(len(results)),
        "n_unmatched":           len(unmatched),
        "per_bin_matched":       pd.Series(
            [r["maf_bin"] for r in results]).value_counts().to_dict(),
        "per_bin_aging":         aging_match["target_bin"].value_counts().to_dict(),
        "source":                "1000G GRCh38 high-coverage phased panel (EBI)",
        "seed":                  args.seed,
    }
    Path(args.log).write_text(json.dumps(log, indent=2, default=str))

    print(f"\nDone. {len(results)} matched, {len(unmatched)} unmatched", flush=True)
    print(f"  → {out_path}", flush=True)
    print(f"  → {args.log}", flush=True)


if __name__ == "__main__":
    main()
