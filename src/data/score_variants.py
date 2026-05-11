"""
Score variants through AlphaGenome. Resumable.

This is a thin wrapper that imports v1's scoring logic (lifted into
src.scoring_helpers) and applies it to either aging or control variants.

Usage:
    python src/data/score_variants.py \\
        --input data/processed/controls.csv \\
        --output-dir data/alphagenome_scores/controls
"""
import argparse
import signal
import time
from pathlib import Path

import pandas as pd

from alphagenome.models import dna_client
from alphagenome.data import genome


class _VariantTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _VariantTimeout()

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.scoring_helpers import classify_tissue, extract_scores_from_result


def score_one(model, scorers, rsid, chromosome, position, ref, alt,
              output_dir, max_retries=3):
    out_path = Path(output_dir) / f"{rsid}.parquet"
    if out_path.exists():
        return "cached"

    valid = set("ACGTN")
    if (not all(c in valid for c in str(ref)) or
            not all(c in valid for c in str(alt))):
        return "invalid_alleles"

    variant = genome.Variant(
        chromosome=chromosome,
        position=int(position),
        reference_bases=ref,
        alternate_bases=alt,
    )
    half = dna_client.SEQUENCE_LENGTH_1MB // 2
    interval = genome.Interval(
        chromosome=chromosome,
        start=int(position) - half,
        end=int(position) + half,
    )

    for attempt in range(max_retries):
        try:
            # Hard wall-clock timeout per attempt. AlphaGenome client
            # occasionally hangs on gRPC reads with no internal timeout —
            # the signal.alarm prevents a single call from freezing the
            # whole run indefinitely.
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(120)  # 120s per attempt; typical call is 30-60s
            try:
                results = model.score_variant(
                    interval=interval, variant=variant, variant_scorers=scorers
                )
            finally:
                signal.alarm(0)

            rows = []
            for idx, adata in enumerate(results):
                rows.extend(extract_scores_from_result(adata, idx, rsid))
            if not rows:
                return "empty"
            df = pd.DataFrame(rows)
            df["tissue_category"] = df["biosample_name"].apply(classify_tissue)
            df.to_parquet(out_path, index=False)
            return "ok"
        except _VariantTimeout:
            print(f"    [timeout] {rsid} attempt {attempt+1}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return "timeout"
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "UNAVAILABLE" in msg:
                wait = (attempt + 1) * 30
                time.sleep(wait)
                continue
            if "INVALID_ARGUMENT" in msg:
                return f"invalid_arg: {msg[:80]}"
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return f"error: {msg[:100]}"
    return "retries_exhausted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--api-key-file", default="env/api_key.txt")
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.limit > 0:
        df = df.head(args.limit)

    api_key = Path(args.api_key_file).read_text().strip()
    model = dna_client.create(api_key)
    scorers = dna_client.variant_scorers_lib.get_recommended_scorers(
        dna_client.Organism.HOMO_SAPIENS
    )

    already = sum(1 for f in out_dir.glob("*.parquet"))
    print(f"Scoring {len(df)} variants → {out_dir} "
          f"({already} already cached)", flush=True)

    failed, ok, cached = [], 0, 0
    t0 = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        status = score_one(
            model, scorers,
            rsid=row.rsid, chromosome=row.chromosome, position=row.position,
            ref=row.ref_allele, alt=row.alt_allele,
            output_dir=out_dir,
        )
        if status == "ok":
            ok += 1
        elif status == "cached":
            cached += 1
        else:
            failed.append((row.rsid, status))

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta_min = (elapsed / (i + 1)) * (len(df) - i - 1) / 60
            print(f"  [{i+1}/{len(df)}] ok={ok} cached={cached} "
                  f"fail={len(failed)}  ETA ~{eta_min:.0f}min", flush=True)
        time.sleep(args.delay)

    print(f"\nDone. ok={ok} cached={cached} fail={len(failed)}")
    if failed:
        with open(out_dir / "failed.txt", "w") as f:
            for rsid, status in failed:
                f.write(f"{rsid}\t{status}\n")
        print(f"  Failed log: {out_dir}/failed.txt")


if __name__ == "__main__":
    main()
