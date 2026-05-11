#!/usr/bin/env bash
#
# Run once data/processed/controls.csv is populated with MAF-matched controls
# from collect_controls_v2.py. Drives the rest of the pipeline:
#
#   1. Score new controls through AlphaGenome       (multi-hour, resumable)
#   2. Rebuild control feature matrix                (minutes)
#   3. Retrain binary + multi-class with 1:1 design  (~15 min)
#   4. Rebuild webapp gallery                        (seconds)
#   5. Snapshot v2 metrics then archive as v2.0      (seconds)
#
# Usage (from repo root):
#   bash scripts/rebuild_after_controls.sh

set -euo pipefail
cd "$(dirname "$0")/.."

source activate aging-atlas

# Sanity: controls.csv must exist
if [[ ! -s data/processed/controls.csv ]]; then
  echo "FATAL: data/processed/controls.csv not found — run collect_controls_v2.py first"
  exit 1
fi

N_CONTROLS=$(tail -n +2 data/processed/controls.csv | wc -l | tr -d ' ')
echo "Found $N_CONTROLS controls"

# Snapshot current v2 artifacts so we can compare before/after
echo ""
echo "=== Snapshotting current v2 metrics as v2.0 (imbalanced, 500 v1 controls)"
mkdir -p results/snapshots
if [[ -f results/metrics.json ]]; then
  cp results/metrics.json "results/snapshots/metrics_v2.0_$(date +%Y%m%d).json"
fi

# Move old v1 control scores aside so build_features only sees new ones
echo ""
echo "=== Swapping control scores directory (v1 → aging_scores/controls_v1_backup)"
if [[ -d data/alphagenome_scores/controls ]] && \
   [[ ! -d data/alphagenome_scores/controls_v1_backup ]] && \
   [[ -z "$(ls data/alphagenome_scores/controls_v1_backup 2>/dev/null)" ]]; then
  # Only move if not already moved
  n_old=$(ls data/alphagenome_scores/controls/ 2>/dev/null | wc -l | tr -d ' ')
  if [[ "$n_old" -gt 0 ]] && [[ "$n_old" -lt 2000 ]]; then
    # Heuristic: old v1 set is 500; new set will be ~5000
    echo "  Moving $n_old v1 control scores to controls_v1_backup/"
    mv data/alphagenome_scores/controls data/alphagenome_scores/controls_v1_backup
    mkdir -p data/alphagenome_scores/controls
  fi
fi

echo ""
echo "=== Step 1/5: AlphaGenome scoring of MAF-matched controls"
echo "    (this is the long one — resumable, safe to Ctrl-C and rerun)"
python src/data/score_variants.py \
  --input data/processed/controls.csv \
  --output-dir data/alphagenome_scores/controls \
  --delay 1.0

n_scored=$(ls data/alphagenome_scores/controls/ | wc -l | tr -d ' ')
echo "  scored $n_scored / $N_CONTROLS controls"

echo ""
echo "=== Step 2/5: Rebuild control feature matrix"
python src/features/build_features.py \
  --scores-dir data/alphagenome_scores/controls \
  --out data/processed/features_controls.parquet \
  --label control

echo ""
echo "=== Step 3/5: Regression tests (leakage guards)"
python -m pytest tests/ -v

echo ""
echo "=== Step 4/5: Retrain (1,000-permutation test)"
python src/models/train.py --n-permutations 1000

echo ""
echo "=== Step 5/5: Rebuild webapp gallery"
python src/inference/build_gallery.py

echo ""
echo "All done. Key artifacts:"
echo "  results/metrics.json                 (v2.1 balanced design)"
echo "  results/snapshots/metrics_v2.0_*.json (prior v2.0 for comparison)"
echo "  models/binary_rf.joblib              (retrained)"
echo "  models/multiclass_rf.joblib          (retrained)"
echo "  webapp/gallery.json                  (rebuilt)"
echo ""
echo "To compare v2.0 vs v2.1:"
echo "  python scripts/compare_runs.py"
