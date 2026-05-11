#!/usr/bin/env bash
#
# Run the entire AlphaGenome × Aging Variant Study pipeline end-to-end.
#
# Assumes:
#   - conda env 'aging-atlas' with environment.yml installed
#   - env/api_key.txt contains a valid AlphaGenome API key
#   - data/processed/aging_variants.csv is present
#   - data/alphagenome_scores/aging/ contains pre-scored aging parquets
#
# For a fresh run without cached scores, step 1 (control collection) and
# step 2 (AlphaGenome scoring of controls) can each take several hours.
# They are both resumable.

set -euo pipefail
cd "$(dirname "$0")"

source activate aging-atlas

echo "=========================================================="
echo "Step 1/7: MAF-matched control selection (Ensembl REST)"
echo "=========================================================="
if [[ ! -s data/processed/controls.csv ]]; then
  python src/data/collect_controls.py --workers 8 --save-every 50
else
  echo "  data/processed/controls.csv exists; skipping"
fi

echo "=========================================================="
echo "Step 2/7: AlphaGenome scoring of control variants"
echo "=========================================================="
python src/data/score_variants.py \
  --input data/processed/controls.csv \
  --output-dir data/alphagenome_scores/controls \
  --delay 1.0

echo "=========================================================="
echo "Step 3/7: Feature aggregation — aging + controls"
echo "=========================================================="
python src/features/build_features.py \
  --scores-dir data/alphagenome_scores/aging \
  --out data/processed/features_aging.parquet \
  --label aging
python src/features/build_features.py \
  --scores-dir data/alphagenome_scores/controls \
  --out data/processed/features_controls.parquet \
  --label control

echo "=========================================================="
echo "Step 4/7: Gene group assignment (for GroupKFold)"
echo "=========================================================="
python src/data/assign_gene_groups.py \
  --variants data/processed/aging_variants.csv \
  --vep data/processed/vep_annotations.csv \
  --out data/processed/gene_groups.csv

echo "=========================================================="
echo "Step 5/7: Regression tests (leakage guards)"
echo "=========================================================="
python -m pytest tests/ -v

echo "=========================================================="
echo "Step 6/7: Train binary + multi-class classifiers"
echo "=========================================================="
python src/models/train.py --n-permutations 1000

echo "=========================================================="
echo "Step 7/7: Build webapp gallery + launch API"
echo "=========================================================="
python src/inference/build_gallery.py

echo ""
echo "Pipeline complete. To serve:"
echo "  uvicorn src.inference.api:app --host 0.0.0.0 --port 8000"
echo ""
echo "Then open http://localhost:8000/"
