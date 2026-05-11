# Methodology — AlphaGenome × Aging Variant Study

## Summary

We ask whether aging-associated genetic variants have distinguishable regulatory signatures compared to carefully matched non-aging variants, using AlphaGenome (pretrained) as the feature extractor and a small Random Forest as the decision layer. This document is the detailed methodology that the one-page `docs/pipeline.html` references.

---

## 1. The question (stated carefully)

**Not:** "Can we predict biological aging from genomic sequence?"

**Actually:** "Given a variant and AlphaGenome's prediction of its regulatory effect across tissues, can a small classifier distinguish variants drawn from aging/longevity cohort studies from MAF-matched variants not selected for aging?"

These are very different questions. The first is a medical claim; the second is a statistical claim about two variant pools and a pretrained model's output.

---

## 2. Variant inventory

### Aging variants (5,211)

Imported verbatim from v1. Six sources combined:

| Source | N | Selection |
|---|---:|---|
| GWAS Catalog | 4,983 | Trait terms: longevity, aging, age-related disease, telomere length |
| LongevityMap | 183 | SNPs significantly associated with longevity |
| Curated literature | 45 | Landmark papers (APOE, FOXO3, TERT, etc.) |
| HALL DB (gene regions) | via GWAS overlap | Curated aging SNPs |
| Open Genes (2,402 genes) | via GWAS overlap | Aging-related gene bodies |
| CellAge (866 genes) | via GWAS overlap | Senescence-related gene bodies |

Trait categories (used for the multi-class task):
cardiovascular, cancer, telomere length, metabolic, sensory aging, longevity, inflammation, musculoskeletal, other aging, organ decline, Alzheimer's, biological aging, Parkinson's, epigenetic aging.

For the multi-class task we keep only classes with ≥30 samples (drops Parkinson's and epigenetic aging).

### Control variants (target ~5,000, 1:1 with aging)

Each aging variant gets one matched control, selected via
`src/data/collect_controls_v2.py`:

1. **MAF extraction.** Each aging variant has a gnomAD AF from VEP. Convert to MAF = min(AF, 1-AF).
2. **MAF bins.** `<0.1%`, `0.1-1%`, `1-5%`, `5-10%`, `10-50%`. Aging variant distribution is heavily in `10-50%` (50%) and converted `>50%` (23%, collapsed into `10-50%` after MAF conversion).
3. **Remote tabix fetch.** `pysam.VariantFile` opens the 1000 Genomes GRCh38 high-coverage phased panel directly over HTTPS:
   ```
   https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/
     1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/
     1kGP_high_coverage_Illumina.chr{N}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz
   ```
   Range requests fetch only the bytes for the target region (±50–1,000 kb, expanding if needed).
4. **MAF from INFO/AF.** The 1000G VCF records have `AF` in the INFO field, so no second lookup is needed.
5. **Match criterion.** Same MAF bin as target. Biallelic SNV (single-base ref and alt). Not in aging rsID set. Not at an aging-variant position (covers aging variants without rsIDs).
6. **Pick.** Random among qualifying matches.

**What this achieves:** matched on chromosome (by construction), genomic neighborhood (same window), MAF bin (by filter), biallelic SNV. Not matched on: functional consequence, ancestry-specific frequencies, trait-negative status.

**Honesty note:** the smoke test on 20 variants showed 100% match rate, per-bin distribution matching the aging bin distribution exactly. Some aging variants will still be unmatched at the full scale — reported in `data/processed/control_selection_log.json`.

**Why 1000G GRCh38 and not Ensembl REST.** We originally tried Ensembl `/overlap/region/variation` + batch POST `/variation/human`. The REST responses for large (±500 kb) windows are multi-megabyte JSON payloads that took 30–500 s per variant with frequent timeouts. The tabix-indexed 1000G VCF returns 3,000–4,000 records in ~1 s per variant via HTTP range requests. Same data, 200× faster.

**Assembly match.** The aging variant positions are hg38 / GRCh38 (verified via VEP annotations and direct position lookup: e.g. rs4970418 at chr1:983,237 matches GRCh38, not GRCh37 where it sits at chr1:1,047,754). The 1000G high-coverage 2022 release is GRCh38-aligned, so no lift-over is needed. v1's 500 Ensembl-random controls are discarded in v2.1 — they were on GRCh37 of unknown-MAF variants, which is exactly the kind of silent mismatch that motivated this redesign.

---

## 3. Feature extraction

### Per-variant AlphaGenome inference

For each variant, a single call to AlphaGenome with the recommended scorers returns 19 AnnData objects covering 5 main output types (RNA_SEQ, ATAC, DNASE, CHIP_HISTONE, CHIP_TF) plus their `_ACTIVE` variants, CAGE, PROCAP, splicing, and polyadenylation.

Input interval: ±500 kb around variant (1 Mb total).

Raw output, once flattened, is ~26,000 prediction rows per variant (715 biosamples × output types × a small gene set for gene-level scorers).

### Tissue categorization

Each AlphaGenome biosample gets mapped to one of 10 coarse tissue categories by keyword matching on the biosample name. Examples: "cortex" → brain; "ventricle" → heart; "skeletal muscle" → muscle. Biosamples that don't match any keyword map to `other` and are ignored in the tissue-level features (but still contribute to globals).

### Feature aggregation (115 features per variant)

Stable summaries chosen to preserve tissue × output-type structure without blowing up feature count:

```
for each output_type in {RNA_SEQ, ATAC, DNASE, CHIP_HISTONE, CHIP_TF}:
    for each tissue in {brain, heart, liver, blood, kidney,
                        muscle, adipose, skin, colon, lung}:
        f"{ot}__{tissue}__max"       — max |score|, signed
        f"{ot}__{tissue}__mean_abs"  — mean |score|
    f"{ot}__global__max"      — max |score| over all biosamples
    f"{ot}__global__mean_abs" — mean |score| over all biosamples
    f"{ot}__global__std"      — std of raw scores
```

Total = 5 × (10 × 2 + 3) = 115.

Implementation: `src/features/build_features.py:build_variant_features`.

**Why these summaries:**
- `max(|x|)` preserves the strongest effect direction; important for regulatory disruption
- `mean(|x|)` captures overall magnitude of effect across biosamples
- `std` captures variance/noise per output type
- `global` features provide cross-tissue summaries for outputs that don't map cleanly to our 10 tissue categories

Alternatives considered and rejected:
- **Raw 26k × N matrix with PCA:** harder to inspect, learned projections add variance
- **Per-biosample features (~3,500 features):** too sparse, overfitting risk with only ~10k variants
- **Learned tissue embeddings:** would require a second training step; overkill

---

## 4. Splits and validation

### Gene-grouped train/test

Each variant gets a gene group (VEP `gene_symbol` > nearest-gene > 100-kb position bucket as fallback). `GroupShuffleSplit` carves a 20% held-out test set where no gene group crosses the boundary. `GroupKFold(5)` runs on the remaining 80% for CV.

### Why this matters

If a gene has 20 aging variants in tight LD (e.g. APOE region), their AlphaGenome features are correlated. Naive K-fold can put 15 of them in train and 5 in test, where the test 5 are effectively memorizable. GroupKFold forces all 20 into the same fold.

### Preprocessing inside the Pipeline

`SimpleImputer(fill_value=0.0)` + `StandardScaler` sit in the sklearn Pipeline alongside the classifier. When CV or `permutation_test_score` fits the Pipeline, preprocessing is refit per training fold, never on the held-out fold.

### Leakage audit

We also run a naive (non-grouped) stratified K-fold and report the AUC difference. This number tells a reader how much of v1's metrics would have been inflation from LD/gene leakage.

---

## 5. Metrics

### Binary task (aging vs control)

| Metric | Purpose |
|---|---|
| **Balanced accuracy** | Headline. Insensitive to class ratio. |
| **ROC-AUC** | Threshold-independent discrimination. |
| **PR-AUC** | Informative under class rarity. Balance is 1:1 here, but reporting it keeps us honest under future edits. |
| **Brier score** | Mean squared error of predicted probabilities vs outcomes. Lower is better. Measures calibration. |
| **F1** | Harmonic mean of precision and recall for the positive class (aging). |
| **Raw accuracy** | Reported alongside, never headline-only. |
| **Permutation p-value** (n=1000) | Shuffle labels within grouped CV scheme, recompute AUC, fraction ≥ observed is p. |
| **Grouped vs naive CV AUC delta** | Leakage audit. |

### Multi-class task (trait category, aging-only)

| Metric | Purpose |
|---|---|
| **Accuracy** | Plain accuracy against multi-class labels. |
| **Balanced accuracy** | Accuracy averaged across classes — handles class imbalance. |
| **Majority-class baseline** | Accuracy if we always predict the largest class. |
| **Random baseline (1/K)** | Accuracy if we predict uniformly. |

Confusion matrices are saved as figures for both tasks.

### What we do NOT report as headline

- **Raw accuracy on an imbalanced task.** If the task ever drifts from 1:1, switch to balanced accuracy.
- **CV metric as "final".** Test set on held-out genes is the final number.
- **Naive-CV AUC as "the AUC".** Only grouped CV is valid; the naive number exists as an audit artifact.

---

## 6. Classifier configuration

### Random Forest (primary)

```python
RandomForestClassifier(
    n_estimators=300,       # 400 for multi-class
    max_depth=10,           # modest; we have ~10k samples
    class_weight='balanced', # cosmetic at 1:1 but defensive
    random_state=42,
    n_jobs=-1,
)
```

### Logistic Regression (secondary, available via `make_logreg_pipeline`)

Same pipeline, linear classifier. Kept as an option for calibration comparisons.

### Why RF?

- Handles NaN-imputed features gracefully
- Non-linear interactions across feature space are plausible (tissue × output type combinations)
- Feature importances are informative for the "top contributing features" UI panel
- ~10k samples × 115 features is a size where RF usually beats LR

**We do not claim RF is the best model.** It's a reasonable default for a small tabular problem; the pipeline is agnostic to classifier choice.

---

## 7. Inference at serving time

The webapp's live path reuses the exact same feature aggregation as training, so cached and live predictions are interchangeable.

```
rsID → Ensembl /variation/human/{rsid}
     → (chromosome, position, ref, alt)
     → AlphaGenome score_variant  (~30-60s)
     → build_variant_features()   (identical to training)
     → Pipeline.predict_proba     (binary + multi-class)
     → JSON response:
         - variant coordinates
         - P(aging)
         - P(trait) per category
         - tissue × output heatmap (max signed |score| per cell)
         - top 10 features by importance × |scaled value|
```

A `cache_scores_dir` lookup skips the AlphaGenome call for variants we've already scored, which makes the gallery instant.

---

## 8. Reproducibility

- `random_state=42` everywhere
- `environment.yml` pins AlphaGenome v0.6.1
- All CSV and parquet artifacts versioned in `data/processed/`
- Unit tests in `tests/test_no_leakage.py` codify the v1 → v2 methodology invariants
- Model artifacts in `models/*.joblib` include feature column names + class list

---

## 9. Known caveats (repeated with emphasis)

- **Heterogeneous labels.** "Aging" is a six-source union; subclasses differ more than they agree.
- **MAF source bias.** 1000 Genomes / gnomAD frequencies are primarily European-ancestry-weighted.
- **Trait-negative not verified.** Controls match on MAF/chromosome/neighborhood but may incidentally carry trait associations.
- **AlphaGenome outputs are model predictions, not measurements.**
- **Classifier is a shallow decision layer.** Any signal reflects AlphaGenome's priors + our aggregation + modest statistical discrimination, not a new discovery about aging biology.

If the headline metric surprises a reader, the intent of this document is that they can follow the citations to figure out exactly why.
