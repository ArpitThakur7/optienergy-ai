# Model card: OptiEnergy AI

**Task:** next-hour household active energy in kWh. **Selected family:** Histogram boosting. **Deployment seed:** 7. The seed-specific artifact and paired uncertainty analysis differ from the seed-averaged model-family score.

**Intended use:** education, reproducible optimization research, and historical one-step forecasting on the supplied household. **Evaluation:** three outer calendar validation folds; separate inner checkpoint window; 24-hour embargo; final 20% of elapsed time excluded from all configuration/checkpoint selection. Each method is scored on common observed targets. Scalers fit only inner training data. The boosted-tree model disables random internal early stopping.

**Performance:** see FINDINGS.md, forecasting_leaderboard.csv and holdout_runs.csv. Fold SD measures temporal validation variation. Seed SD measures the selected stochastic model's variability; a deterministic one-seed result has no estimated seed SD. Block-bootstrap uncertainty concerns fixed-seed errors on one held-out period.

**Limits:** one household; missing intervals removed from common support; no weather or occupancy; no external household validation; possible temporal distribution changes; no calibrated individual forecast intervals; no guarantee beyond this historical setting; no causal feature attribution. One-step evaluation may use observed prior test hours, as available at each forecast origin. Recursive multi-horizon deployment needs its own evaluation.

**Physical constraints:** outputs are not clipped during comparison. Negative forecasts indicate model error. They should not be interpreted as physically valid consumption.

**Operational notes:** CPU-only experiments, deterministic seeds and declared budgets. The trusted model file is generated locally and omitted from GitHub. Pickle/joblib artifacts should only be loaded from trusted sources after executing the notebook's class-definition cells.
