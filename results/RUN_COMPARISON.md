# Run comparison: v2.0_20260418 vs current

Snapshot: `results/snapshots/...` | Current: `results/metrics.json`

## Binary task — aging vs control

| Metric | v2.0_20260418 | current |
|---|---:|---:|
| Train N | 4705 | 8324 |
| Test N | 1199 | 2134 |
| Train positives (aging) | 4300 | 4299 |
| Gene groups (train) | 2941 | 2541 |

| Metric | v2.0_20260418 (test) | current (test) |
|---|---:|---:|
| Balanced accuracy | 0.664 | 0.574 |
| ROC-AUC | 0.838 | 0.601 |
| PR-AUC | 0.982 | 0.635 |
| Brier | 0.071 | 0.240 |
| F1 (aging) | 0.953 | 0.481 |
| Raw accuracy | 0.913 | 0.567 |

| Metric | v2.0_20260418 | current |
|---|---:|---:|
| Permutation p | 0.0010 | 0.0010 |
| Naive CV AUC | 0.859 | 0.583 |
| Grouped CV AUC | 0.851 | 0.580 |
| Leakage delta AUC | 0.008 | 0.003 |

## Multi-class task — trait category

| Metric | v2.0_20260418 | current |
|---|---:|---:|
| N classes | 12 | 12 |
| Test accuracy | 0.362 | 0.362 |
| Test balanced accuracy | 0.096 | 0.096 |
| Majority baseline acc | 0.502 | 0.502 |
| Random baseline acc | 0.083 | 0.083 |

## Interpretation

- Balanced accuracy: **0.664 → 0.574** (-0.091)
- ROC-AUC:           **0.838 → 0.601** (-0.238)
- Class ratio (ctrl:aging) train: 0.09 → 0.94
