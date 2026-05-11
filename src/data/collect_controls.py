"""
Collect MAF-matched control variants for the AlphaGenome × Aging Variant Study.

Strategy:
    For each aging variant, query Ensembl /overlap/region for nearby variants
    (+/- 50-500kb), then BATCH query their 1000-Genomes MAF via POST
    /variation/human. Pick one with matching MAF bin, excluding aging rsIDs.
    This gives controls matched on: chromosome, genomic neighborhood, MAF bin.

Ensembl endpoints used:
    GET  /overlap/region/human/:region?feature=variation  → rsIDs only
    POST /variation/human                                  → MAF (batch, up to 200 rsIDs)

Output:
    data/processed/controls.csv — per-aging-variant matched control
    data/processed/control_selection_log.json — provenance + per-bin counts

Resumable.
"""
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


ENSEMBL_REST = "https://rest.ensembl.org"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
BATCH_SIZE = 200  # Ensembl POST /variation/human accepts up to 200 ids

# MAF bins. Note: MAF = min(af, 1-af), so MAF is in [0, 0.5].
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


def _request(method, url, *, retries=3, backoff=2.0, **kwargs):
    for attempt in range(retries):
        try:
            r = requests.request(method, url, timeout=30, **kwargs)
            if r.status_code == 429:
                wait = backoff ** (attempt + 1)
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                return None
            time.sleep(backoff ** (attempt + 1))
    return None


def query_region_variants(chromosome, start, end):
    """Return list of candidate variants (SNV rsIDs) in region."""
    chrom = chromosome.replace("chr", "")
    url = f"{ENSEMBL_REST}/overlap/region/human/{chrom}:{start}-{end}"
    data = _request("GET", url, headers=HEADERS, params={"feature": "variation"})
    if not data:
        return []
    out = []
    for v in data:
        rsid = v.get("id", "")
        if not rsid.startswith("rs"):
            continue
        alleles = v.get("alleles") or []
        # Keep only biallelic SNVs with single-base alleles
        if len(alleles) != 2:
            continue
        if any(len(a) != 1 for a in alleles):
            continue
        out.append({
            "rsid": rsid,
            "chromosome": chromosome,
            "position": int(v.get("start", 0)),
            "ref_allele": alleles[0],
            "alt_allele": alleles[1],
            "consequence_type": v.get("consequence_type", ""),
        })
    return out


def batch_query_mafs(rsids):
    """POST /variation/human for up to BATCH_SIZE rsIDs. Returns {rsid: maf}."""
    if not rsids:
        return {}
    out = {}
    for i in range(0, len(rsids), BATCH_SIZE):
        chunk = rsids[i:i + BATCH_SIZE]
        body = {"ids": chunk, "pops": 1}
        data = _request("POST", f"{ENSEMBL_REST}/variation/human",
                        headers=HEADERS, json=body)
        if not data:
            continue
        for rsid, info in data.items():
            # minor_allele_freq is the top-level 1000G MAF (global)
            maf = info.get("minor_allele_freq")
            if maf is not None:
                try:
                    out[rsid] = float(maf)
                except (TypeError, ValueError):
                    pass
    return out


def pick_matched_control(aging_row, aging_rsids, sleep=0.0,
                         windows=(50_000, 200_000, 1_000_000),
                         max_batches_per_window=3):
    """For one aging variant, find and return a matched control dict, or None.

    For each window size, query the region once, then batch-lookup MAF in
    chunks of BATCH_SIZE (up to `max_batches_per_window` chunks) until we
    find a MAF-bin match or exhaust the window.
    """
    target_maf = to_maf(aging_row.get("gnomad_af"))
    if target_maf is None:
        return None
    target_bin = maf_bin(target_maf)
    if target_bin is None:
        return None

    chrom = aging_row["chromosome"]
    pos = int(aging_row["position"])

    for w in windows:
        start = max(1, pos - w)
        end = pos + w
        candidates = query_region_variants(chrom, start, end)
        candidates = [c for c in candidates if c["rsid"] not in aging_rsids]
        if not candidates:
            if sleep:
                time.sleep(sleep)
            continue

        random.shuffle(candidates)

        # Try up to max_batches_per_window chunks of BATCH_SIZE
        for batch_i in range(max_batches_per_window):
            chunk = candidates[batch_i * BATCH_SIZE:(batch_i + 1) * BATCH_SIZE]
            if not chunk:
                break
            rsids = [c["rsid"] for c in chunk]
            mafs = batch_query_mafs(rsids)
            if sleep:
                time.sleep(sleep)

            matched = []
            for c in chunk:
                m = mafs.get(c["rsid"])
                if m is None:
                    continue
                m = to_maf(m)
                if maf_bin(m) == target_bin:
                    c2 = dict(c)
                    c2["gnomad_af"] = m
                    c2["maf_bin"] = target_bin
                    c2["matched_to"] = aging_row["rsid"]
                    c2["matched_distance_bp"] = abs(c["position"] - pos)
                    matched.append(c2)
            if matched:
                return random.choice(matched)

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aging-csv", default="data/processed/aging_variants.csv")
    ap.add_argument("--vep-csv",   default="data/processed/vep_annotations.csv")
    ap.add_argument("--out",       default="data/processed/controls.csv")
    ap.add_argument("--log",       default="data/processed/control_selection_log.json")
    ap.add_argument("--sleep",     type=float, default=0.1)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--limit",     type=int, default=0)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent Ensembl queries (Ensembl allows ~15 rps)")
    args = ap.parse_args()

    random.seed(args.seed)

    aging = pd.read_csv(args.aging_csv)
    vep = pd.read_csv(args.vep_csv, usecols=["rsid", "gnomad_af"])
    aging = aging.merge(vep, on="rsid", how="left")
    aging_rsids = set(aging["rsid"])
    print(f"Loaded {len(aging)} aging variants "
          f"({aging['gnomad_af'].notna().sum()} with gnomad_af)", flush=True)

    aging_match = aging[aging["gnomad_af"].notna()].copy()
    aging_match["target_maf"] = aging_match["gnomad_af"].apply(to_maf)
    aging_match["target_bin"] = aging_match["target_maf"].apply(maf_bin)
    aging_match = aging_match.dropna(subset=["target_bin"])
    print(f"  Matchable (have MAF bin): {len(aging_match)}", flush=True)

    if args.limit > 0:
        aging_match = aging_match.head(args.limit)
        print(f"  Limited to first {args.limit}", flush=True)

    out_path = Path(args.out)
    if out_path.exists() and out_path.stat().st_size > 0:
        existing = pd.read_csv(out_path)
        matched_rsids = set(existing["matched_to"])
        results = existing.to_dict("records")
        print(f"Resuming from {out_path}: {len(results)} already done", flush=True)
    else:
        matched_rsids = set()
        results = []

    # Parallelize: each aging variant is an independent job doing network I/O
    # against Ensembl. ThreadPoolExecutor with ~10 workers respects Ensembl's
    # per-IP rate limit while hiding per-call latency.
    unmatched = []
    total_todo = len(aging_match)
    to_process = [row for _, row in aging_match.iterrows()
                  if row["rsid"] not in matched_rsids]
    t0 = time.time()

    def work(row):
        t_var = time.time()
        ctrl = pick_matched_control(row, aging_rsids, sleep=args.sleep)
        return row, ctrl, time.time() - t_var

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(work, row): row for row in to_process}
        for n, fut in enumerate(as_completed(futures), start=1):
            row, ctrl, dt = fut.result()
            status = "OK" if ctrl is not None else "MISS"
            if ctrl is not None:
                ctrl["aging_bin"] = row["target_bin"]
                results.append(ctrl)
            else:
                unmatched.append(row["rsid"])

            if n % 10 == 0 or n == len(to_process):
                n_done = len(results)
                n_miss = len(unmatched)
                rate = n_done / max(1, (n_done + n_miss))
                elapsed = time.time() - t0
                per = elapsed / n
                eta_min = per * (len(to_process) - n) / 60
                print(f"[{n}/{len(to_process)}] matched={n_done} "
                      f"unmatched={n_miss} hit_rate={rate:.1%} "
                      f"per={per:.1f}s ETA ~{eta_min:.0f}min",
                      flush=True)
                pd.DataFrame(results).to_csv(out_path, index=False)
            else:
                # per-variant status line (terse)
                print(f"  [{n}] {row['rsid']} bin={row['target_bin']} "
                      f"{status} {dt:.0f}s", flush=True)

    pd.DataFrame(results).to_csv(out_path, index=False)

    log = {
        "n_aging_input":         int(len(aging)),
        "n_aging_matchable":     int(total),
        "n_controls_matched":    int(len(results)),
        "n_unmatched":           len(unmatched),
        "per_bin_matched":       pd.Series(
            [r["maf_bin"] for r in results]).value_counts().to_dict(),
        "per_bin_aging":         aging_match["target_bin"].value_counts().to_dict(),
        "seed":                  args.seed,
    }
    Path(args.log).write_text(json.dumps(log, indent=2, default=str))

    print(f"\nDone. {len(results)} matched, {len(unmatched)} unmatched", flush=True)
    print(f"  → {out_path}", flush=True)
    print(f"  → {args.log}", flush=True)


if __name__ == "__main__":
    main()
