"""
End-to-end inference for a single variant.

Pipeline:
    1. Resolve rsID (or chr:pos:ref>alt) to (chromosome, position, ref, alt)
       via Ensembl REST.
    2. Call AlphaGenome API for scores across all recommended scorers.
    3. Aggregate scores into the 115-d feature vector used at training time.
    4. Load trained binary + multi-class classifiers.
    5. Return prediction + per-feature contributions + per-tissue heatmap data.

Usage (CLI):
    python src/inference/predict_variant.py --rsid rs7412
    python src/inference/predict_variant.py --variant chr19:44908822:C:T

Importable:
    from src.inference.predict_variant import predict
    result = predict(rsid="rs7412")
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from src.features.build_features import build_variant_features, OUTPUT_TYPES, TISSUES

ENSEMBL_REST = "https://rest.ensembl.org"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}


def resolve_rsid(rsid: str):
    """Resolve rsID → (chromosome, position, ref, alt). Returns None on failure."""
    url = f"{ENSEMBL_REST}/variation/human/{rsid}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    mappings = data.get("mappings", [])
    if not mappings:
        return None
    m = mappings[0]
    seq = m.get("seq_region_name", "")
    # Use GRCh38 assembly (AlphaGenome expects hg38)
    for mm in mappings:
        if mm.get("assembly_name") == "GRCh38":
            m = mm
            seq = m.get("seq_region_name", "")
            break
    chrom = f"chr{seq}"
    pos = int(m.get("start", 0))
    alleles = m.get("allele_string", "").split("/")
    if len(alleles) < 2:
        return None
    return {
        "rsid": rsid,
        "chromosome": chrom,
        "position": pos,
        "ref_allele": alleles[0],
        "alt_allele": alleles[1],
    }


def parse_variant_string(s: str):
    """Parse 'chr19:44908822:C:T' or 'chr19:44908822:C>T' into variant dict."""
    s = s.replace(">", ":").replace("-", ":")
    parts = s.split(":")
    if len(parts) != 4:
        return None
    chrom, pos, ref, alt = parts
    if not chrom.startswith("chr"):
        chrom = f"chr{chrom}"
    return {
        "rsid": s,
        "chromosome": chrom,
        "position": int(pos),
        "ref_allele": ref,
        "alt_allele": alt,
    }


def score_with_alphagenome(variant_dict, api_key: str):
    """Call AlphaGenome and return the same long-format DataFrame used in training."""
    from alphagenome.models import dna_client
    from alphagenome.data import genome
    from src.scoring_helpers import classify_tissue, extract_scores_from_result

    model = dna_client.create(api_key)
    scorers = dna_client.variant_scorers_lib.get_recommended_scorers(
        dna_client.Organism.HOMO_SAPIENS
    )

    variant = genome.Variant(
        chromosome=variant_dict["chromosome"],
        position=int(variant_dict["position"]),
        reference_bases=variant_dict["ref_allele"],
        alternate_bases=variant_dict["alt_allele"],
    )
    half = dna_client.SEQUENCE_LENGTH_1MB // 2
    interval = genome.Interval(
        chromosome=variant_dict["chromosome"],
        start=int(variant_dict["position"]) - half,
        end=int(variant_dict["position"]) + half,
    )

    results = model.score_variant(interval=interval, variant=variant,
                                  variant_scorers=scorers)

    rows = []
    for idx, adata in enumerate(results):
        rows.extend(extract_scores_from_result(adata, idx, variant_dict["rsid"]))

    df = pd.DataFrame(rows)
    df["tissue_category"] = df["biosample_name"].apply(classify_tissue)
    return df


def predict(rsid=None, variant_str=None, api_key=None,
            models_dir="models", cache_scores_dir=None):
    """Main entry point. Returns a dict with prediction + explainability."""
    # 1. Resolve / parse
    if rsid:
        var = resolve_rsid(rsid)
        if var is None:
            return {"error": f"Could not resolve rsID {rsid}"}
    elif variant_str:
        var = parse_variant_string(variant_str)
        if var is None:
            return {"error": f"Could not parse variant {variant_str}"}
    else:
        return {"error": "Must provide rsid or variant_str"}

    # 2. Cache check (pre-scored variants reuse frozen scores)
    if cache_scores_dir:
        cache_path = Path(cache_scores_dir) / f"{var['rsid']}.parquet"
        if cache_path.exists():
            scores_df = pd.read_parquet(cache_path)
        else:
            scores_df = None
    else:
        scores_df = None

    # 3. Live AlphaGenome call if no cache
    if scores_df is None:
        if not api_key:
            return {"error": "Live inference requires api_key; no cached scores"}
        scores_df = score_with_alphagenome(var, api_key)

    # 4. Aggregate features
    features = build_variant_features(scores_df)

    # 5. Load classifiers
    models_dir = Path(models_dir)
    binary_bundle = joblib.load(models_dir / "binary_rf.joblib")
    feat_cols = binary_bundle["feature_cols"]
    feat_vec = np.array([features.get(c, np.nan) for c in feat_cols]).reshape(1, -1)

    pipe = binary_bundle["pipeline"]
    classes = binary_bundle["classes"]
    probs = pipe.predict_proba(feat_vec)[0]
    pred = pipe.predict(feat_vec)[0]

    result = {
        "variant": var,
        "binary_prediction": str(pred),
        "binary_probabilities": {str(c): float(p) for c, p in zip(classes, probs)},
        "top_features": _top_feature_contributions(pipe, feat_cols, feat_vec, top_k=10),
        "tissue_heatmap": _tissue_heatmap(scores_df),
    }

    # 6. Multi-class (trait) prediction if classifier is available
    mc_path = models_dir / "multiclass_rf.joblib"
    if mc_path.exists():
        mc_bundle = joblib.load(mc_path)
        mc_pipe = mc_bundle["pipeline"]
        mc_classes = mc_bundle["classes"]
        mc_feat_cols = mc_bundle["feature_cols"]
        mc_vec = np.array([features.get(c, np.nan) for c in mc_feat_cols]).reshape(1, -1)
        mc_probs = mc_pipe.predict_proba(mc_vec)[0]
        mc_pred = mc_pipe.predict(mc_vec)[0]
        result["trait_prediction"] = str(mc_pred)
        result["trait_probabilities"] = {
            str(c): float(p) for c, p in zip(mc_classes, mc_probs)
        }

    return result


def _top_feature_contributions(pipe, feat_cols, feat_vec, top_k=10):
    """Return top-K features by global importance × |feature value|."""
    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return []
    imp = clf.feature_importances_
    # Use scaled value (after imputer + scaler) to be comparable across features
    x_scaled = pipe.named_steps["scale"].transform(
        pipe.named_steps["impute"].transform(feat_vec)
    )[0]
    contrib = imp * np.abs(x_scaled)
    order = np.argsort(contrib)[::-1][:top_k]
    return [
        {"feature": feat_cols[i],
         "value_raw": float(feat_vec[0, i]) if not np.isnan(feat_vec[0, i]) else None,
         "value_scaled": float(x_scaled[i]),
         "importance": float(imp[i]),
         "contribution_score": float(contrib[i])}
        for i in order
    ]


def _tissue_heatmap(scores_df):
    """Return a {output_type: {tissue: score}} dict for visualization."""
    out = {}
    for ot in OUTPUT_TYPES:
        sub = scores_df[scores_df["output_type"] == ot]
        out[ot] = {}
        for t in TISSUES:
            ts = sub[sub["tissue_category"] == t]["raw_score"].values
            valid = ts[~np.isnan(ts)]
            if len(valid):
                # representative: max absolute (preserving sign)
                idx = int(np.argmax(np.abs(valid)))
                out[ot][t] = float(valid[idx])
            else:
                out[ot][t] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsid", default=None)
    ap.add_argument("--variant", default=None,
                    help="chr:pos:ref:alt (e.g. chr19:44908822:C:T)")
    ap.add_argument("--api-key-file", default="env/api_key.txt")
    ap.add_argument("--cache-dir", default="data/alphagenome_scores/aging")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out", default=None, help="Write JSON to this path")
    args = ap.parse_args()

    api_key = None
    key_path = Path(args.api_key_file)
    if key_path.exists():
        api_key = key_path.read_text().strip()

    result = predict(
        rsid=args.rsid,
        variant_str=args.variant,
        api_key=api_key,
        models_dir=args.models_dir,
        cache_scores_dir=args.cache_dir,
    )

    text = json.dumps(result, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
