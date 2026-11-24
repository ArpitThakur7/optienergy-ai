# Measured portfolio findings

The validation-selected forecasting family is **Histogram boosting**. Its final holdout RMSE is **0.4738 kWh**, averaged across its declared seeds. The strongest naive reference selected on cross-validation is **Persistence**, with holdout RMSE **0.5995 kWh**. The relative change is **20.98% lower RMSE** (negative values would mean worse performance).

The final holdout covers **2010-02-13 02:00:00 through 2010-11-26 21:00:00**, with **6,296** common-support observations. Selection was frozen before final scoring. No family is chosen by the test leaderboard.

| Family | CV RMSE | CV fold SD | Holdout RMSE | Seed SD | Holdout MAE | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| Histogram boosting | 0.5222 | 0.0784 | 0.4738 | deterministic | 0.3296 | 1 |
| ANN / Adam | 0.5390 | 0.0784 | 0.4937 | 0.0081 | 0.3448 | 3 |
| Ridge direct | 0.5811 | 0.0879 | 0.5270 | deterministic | 0.3755 | 1 |
| OLS | 0.5813 | 0.0877 | 0.5273 | deterministic | 0.3755 | 1 |
| Ridge / Adam | 0.5813 | 0.0884 | 0.5273 | 0.0011 | 0.3766 | 3 |
| Persistence | 0.6623 | 0.0955 | 0.5995 | deterministic | 0.3933 | 1 |
| Trailing 24h mean | 0.8149 | 0.1643 | 0.7041 | deterministic | 0.5543 | 1 |
| Weekly seasonal | 0.8537 | 0.1437 | 0.8155 | deterministic | 0.5631 | 1 |
| Daily seasonal | 0.9054 | 0.1715 | 0.7830 | deterministic | 0.5274 | 1 |

## Uncertainty is a separate question

For the fixed deployment seed 7, delta RMSE against the cross-validation-selected naive reference is **-0.1258 kWh**. A paired seven-calendar-day moving-block bootstrap gives an approximate 95% percentile interval **[-0.1412, -0.1059] kWh**. Negative differences favor the champion. One- and fourteen-day intervals are exported to disclose block-length sensitivity. This is an interval on an aggregate metric difference, not an individual forecast interval, and does not include training-data uncertainty. Stationarity assumptions and one-household sampling limit interpretation.

## What distinguishes the two studies

The optimization lab compares eight algorithms under fixed within-family settings on a small contiguous demonstration window, with paired initialization and budgets. The forecasting benchmark uses later calendar folds and a separate final period, with an inner checkpoint window distinct from outer validation. It includes model classes with different training procedures and tuning budgets; it is not a controlled optimizer ranking.

## Verification

The notebook executed 40 core tests and 12 additional temporal/uncertainty tests. The forecasting study recorded 69 candidate-fold-seed fits/scores, 13 final evaluations, and 9 ablations. Exact settings and environment are in protocol.json.
