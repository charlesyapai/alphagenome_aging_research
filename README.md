# AlphaGenome × Aging Variant Study

A computational genomics project that uses Google DeepMind's **AlphaGenome** (pretrained, Nature 2026) to predict the regulatory effects of aging-associated genetic variants, and trains a small classifier on top of those predictions to ask:

> Do aging-associated variants have measurably different regulatory signatures than carefully matched control variants?

This is **primarily an inference study**, not a new-model study. The biological signal comes from AlphaGenome; the contribution here is the inference pipeline, matched-control design, and honest methodology for evaluating the aging-vs-non-aging question.

---

## Two models, explicit distinction

| | **AlphaGenome** | **Aging classifier (this repo)** |
|-|-|-|
| Who trained it | Google DeepMind | Charles Yap |
| Trained on | Multi-omic tracks across ~700 biosamples | AlphaGenome-derived features on ~5k variants |
| Role here | Frozen inference — scores each variant | Thin decision layer — aging vs matched control |
| Architecture | Large multi-output neural net | Random Forest / logistic regression |
| Input | 1 Mb DNA window around variant | 115 summary features per variant |
| Output | ~26k predictions × tissue × track | P(aging) + trait class |

The portfolio-worthy engineering here is **the pipeline that glues inference to a defensible evaluation**, not a claim about the classifier itself.

---

## What's different from v1 (Idea Testing 1)

v1 reported 92.1% accuracy. On inspection, three problems inflated this number:

1. **Class imbalance hid performance.** 5,211 aging vs 500 controls → majority-class baseline is 91.2%. The 92.1% number was +0.9 points above a trivial baseline.
2. **Controls were not MAF-matched.** Plan said MAF + gene-distance matching; implementation was chromosome-only. 93% of controls had unknown MAF, likely skewing rare; aging variants are preferentially common (mean MAF ≈ 0.3). The classifier could be learning "common vs rare" rather than "aging vs not".
3. **LD leakage via naive CV.** Many aging variants cluster in the same gene (APOE, TERT, FOXO3…). Random K-fold lets correlated variants cross the train/test boundary.

v2 fixes all three:
1. **1:1 MAF-matched design.** ~5,000 controls, each matched to one aging variant on MAF bin, chromosome, and genomic neighborhood.
2. **Gene-grouped splits.** `GroupKFold` + `GroupShuffleSplit` by gene → no variant in the same gene can appear on both sides of train/test.
3. **Preprocessing inside the Pipeline.** `StandardScaler`/`SimpleImputer` are refit on each training fold; no leakage through scaling statistics.
4. **Held-out test set.** 20% of gene groups are held out entirely, never seen by CV or permutation tests. Final metrics are reported on this set.
5. **Honest metrics.** Balanced accuracy, ROC-AUC, PR-AUC, Brier score (calibration). Raw accuracy is reported but never headlined.
6. **Leakage audit.** We compute both naive CV AUC and grouped CV AUC and publish the delta so the inflation from LD leakage is visible.

---

## Directory layout

```
src/
  data/
    collect_controls.py      # MAF-matched controls via Ensembl REST
    assign_gene_groups.py    # variant → gene for GroupKFold
    score_variants.py        # thin wrapper around AlphaGenome API
  features/
    build_features.py        # per-variant aggregation → 115-d vector
  models/
    pipeline.py              # sklearn Pipeline factory (no-leak)
    train.py                 # grouped CV + held-out test + permutation
  inference/
    predict_variant.py       # rsid → AlphaGenome → features → prediction
    api.py                   # FastAPI server for live inference

data/
  processed/
    aging_variants.csv                   # 5,211 aging variants (copied from v1)
    controls.csv                         # MAF-matched controls (v2)
    vep_annotations.csv                  # VEP consequences (copied from v1)
    gene_groups.csv                      # variant → gene group mapping
    features_aging.parquet               # aggregated features (aging)
    features_controls.parquet            # aggregated features (controls)
  alphagenome_scores/
    aging/                               # 5,404 parquet files (copied from v1)
    controls/                            # new control scores (v2)

models/
  binary_rf.joblib                       # aging vs control
  multiclass_rf.joblib                   # trait category (aging-only)

results/
  metrics.json                           # all numbers reported in docs/
  figures/                               # confusion, calibration, importances

tests/
  test_no_leakage.py                     # regression tests for v1 → v2 fixes

webapp/                                  # gallery + live inference demo
docs/
  pipeline.html                          # single-page full pipeline doc
  methodology.md                         # long-form methodology
```

---

## Quickstart

```bash
conda activate aging-atlas

# 1. Collect MAF-matched controls (hours, Ensembl REST)
python src/data/collect_controls.py

# 2. Score controls through AlphaGenome (hours, AlphaGenome API)
python src/data/score_variants.py --input data/processed/controls.csv \
    --output-dir data/alphagenome_scores/controls

# 3. Build feature matrices (minutes)
python src/features/build_features.py \
    --scores-dir data/alphagenome_scores/aging \
    --out data/processed/features_aging.parquet --label aging
python src/features/build_features.py \
    --scores-dir data/alphagenome_scores/controls \
    --out data/processed/features_controls.parquet --label control

# 4. Assign gene groups for grouped CV
python src/data/assign_gene_groups.py \
    --variants data/processed/aging_variants.csv \
    --vep data/processed/vep_annotations.csv \
    --out data/processed/gene_groups.csv

# 5. Train and evaluate
python src/models/train.py

# 6. Predict for a single variant
python src/inference/predict_variant.py --rsid rs7412

# 7. Serve the webapp (hybrid gallery + live inference)
python src/inference/api.py  # then open webapp/index.html
```

---

## Headline metrics (v2.1)

Binary task (aging vs MAF-matched control, 5,211 : 5,054, **1:1 balanced**, gene-grouped CV, 80/20 by gene):

| Metric | CV (train) | **Test (held-out)** |
|---|---:|---:|
| Balanced accuracy | 0.558 | **0.574** |
| ROC-AUC | 0.580 | **0.601** |
| PR-AUC | 0.614 | 0.635 |
| Brier (calibration) | 0.243 | 0.240 |
| Permutation p (n=1,000) | — | **0.001** (0/1,000 shuffles reached observed AUC) |
| Leakage delta AUC (naive − grouped) | — | **+0.003** |

Multi-class trait (12 classes, aging-only, gene-grouped):

| Metric | Test | Random baseline |
|---|---:|---:|
| Balanced accuracy | **0.096** | 0.083 |

### v2.0 → v2.1 evolution (the honest part)

Two training runs shipped. The first (v2.0) used v1's 500 Ensembl-random controls and reported test AUC 0.838. Once v2.1 swapped in 5,054 MAF-matched controls from the 1000 Genomes high-coverage panel, **test AUC dropped to 0.601**. Same model, same pipeline, same code — the only change was making the controls properly frequency-matched.

| | v2.0 | **v2.1** | Delta |
|---|---:|---:|---:|
| Test ROC-AUC | 0.838 | **0.601** | −0.238 |
| Test balanced accuracy | 0.664 | 0.574 | −0.091 |
| Class ratio (aging : control) | 8.6:1 | **~1:1** | — |
| Controls MAF-matched? | No (chromosome only) | **Yes (MAF bin + neighborhood)** | — |

What this means: **v2.0's 0.838 AUC was mostly the classifier learning "is this variant common or rare?"** — because aging variants are GWAS-discovered (preferentially common, mean MAF ~0.31) and v1 controls were Ensembl-random (93% unknown MAF, likely rare). Once controls are drawn from the same MAF distribution, the real aging-regulatory signal is revealed: a **modest but statistically significant AUC of 0.601** (p = 0.001 by 1,000-permutation test). Leakage delta near zero (+0.003) confirms the pipeline itself is clean — the whole v2.0 → v2.1 gap was the control confound.

This is the actual story the portfolio tells. v1 reported 92% accuracy as if it were a biological finding; v2.1 shows the biology is real but quiet.

See `results/RUN_COMPARISON.md`, `docs/pipeline.html`, and `results/metrics.json` for the full numbers.

---

## Caveats (load-bearing)

- **"Aging variant" is a heterogeneous label.** Sources span GWAS catalog trait associations, longevity cohorts, aging gene regions, and curated literature. A single label does not mean one biology.
- **Controls are population-frequency-matched, not trait-negative-verified.** We require controls to be absent from the aging set and to match MAF/chromosome/neighborhood. We do not verify that they have no trait associations — they might.
- **AlphaGenome's predictions are not ground truth.** They are model outputs trained on specific tracks. Any downstream signal reflects AlphaGenome's priors, not direct measurement.
- **The classifier is small and trained on ~10k variants.** Expect it to be a useful prior for exploration, not a diagnostic tool.
