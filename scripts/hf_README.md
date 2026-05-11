---
title: AlphaGenome × Aging Variant Study
emoji: 🧬
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Inference demo testing whether aging-associated genetic variants have distinct regulatory signatures, via AlphaGenome.
---

# AlphaGenome × Aging Variant Study

An end-to-end inference demo that asks: *do aging-associated genetic variants have measurably distinct regulatory signatures compared to MAF-matched controls?*

- **Foundation model:** Google DeepMind's pretrained **AlphaGenome** (Nature 2026), accessed via hosted API. Predicts regulatory effects of any DNA variant across ~700 biosamples and 5 output types (RNA-seq, ATAC, DNase, ChIP-histone, ChIP-TF).
- **Downstream classifier:** Random Forest on a 115-d AlphaGenome summary, trained with gene-grouped cross-validation, held-out test by gene, 1:1 MAF-matched controls from the 1000 Genomes high-coverage panel.

## How to use

1. **Gallery** — click any precomputed aging variant (APOE ε2/ε4, FOXO3, TERT, etc.) for instant prediction.
2. **Predict** — enter any rsID or `chr:pos:ref:alt`. Live AlphaGenome call (~30–60s).
3. **Compare** — two variants side-by-side with a delta heatmap.

## Honest headline numbers

- Test ROC-AUC: **0.601** (1:1 balanced design)
- Permutation p: **0.001** (1,000 shuffles)
- Brier score: 0.24 (well-calibrated for the AUC level)

A previous run with non-MAF-matched controls reported AUC 0.838 — the drop to 0.601 once frequency is properly controlled is the actual scientific finding. See `docs/pipeline.html` and the GitHub repo for the full audit trail.

## Source code

Full source, methodology, v1 → v2 postmortem, and tests:
https://github.com/charlesyapai/alphagenome_aging_research

## Caveats

- "Aging variant" is a heterogeneous label (GWAS, longevity cohorts, aging-gene regions).
- Controls are MAF-matched, not trait-negative-verified.
- AlphaGenome's predictions are model outputs, not direct measurements.
- The classifier is a small decision layer — a useful prior, not a diagnostic.
