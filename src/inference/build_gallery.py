"""
Build webapp/gallery.json — precomputed predictions for a curated set of
aging variants, for the instant-load gallery in the webapp.

Usage:
    python src/inference/build_gallery.py
"""
import json
from pathlib import Path

import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.inference.predict_variant import predict


# Well-known aging / longevity variants. Each will be rendered as a card.
FEATURED_VARIANTS = [
    {"rsid": "rs7412",        "note": "APOE ε2 — Alzheimer's-protective"},
    {"rsid": "rs429358",      "note": "APOE ε4 — Alzheimer's-risk"},
    {"rsid": "rs2802292",     "note": "FOXO3 — longevity"},
    {"rsid": "rs2736100",     "note": "TERT — telomere length"},
    {"rsid": "rs1800562",     "note": "HFE — hemochromatosis, iron overload"},
    {"rsid": "rs1333049",     "note": "CDKN2B-AS1 — coronary artery disease"},
    {"rsid": "rs10757278",    "note": "CDKN2B-AS1 — CAD, aging"},
    {"rsid": "rs10757274",    "note": "CDKN2B-AS1 — CAD"},
    {"rsid": "rs12513649",    "note": "cardiovascular"},
    {"rsid": "rs1556516",     "note": "cardiovascular"},
]


def main():
    cards = []
    aging = pd.read_csv(ROOT / "data" / "processed" / "aging_variants.csv").set_index("rsid")
    for entry in FEATURED_VARIANTS:
        rsid = entry["rsid"]
        result = predict(
            rsid=rsid,
            cache_scores_dir=str(ROOT / "data" / "alphagenome_scores" / "aging"),
            models_dir=str(ROOT / "models"),
            api_key=None,  # gallery is cache-only; skip ones without scores
        )
        if "error" in result:
            print(f"  skip {rsid}: {result['error']}")
            continue

        meta = aging.loc[rsid] if rsid in aging.index else None
        card = {
            "rsid": rsid,
            "note": entry["note"],
            "trait_category": str(meta["trait_category"]) if meta is not None else None,
            "nearest_gene": str(meta["nearest_gene"]) if meta is not None else None,
            "p_aging": result["binary_probabilities"].get("aging"),
            "binary_prediction": result["binary_prediction"],
            "trait_prediction": result.get("trait_prediction"),
        }
        cards.append(card)
        print(f"  {rsid}: P(aging)={card['p_aging']:.3f}")

    out = ROOT / "webapp" / "gallery.json"
    out.write_text(json.dumps({"variants": cards}, indent=2, default=str))
    print(f"Saved {len(cards)} gallery cards to {out}")


if __name__ == "__main__":
    main()
