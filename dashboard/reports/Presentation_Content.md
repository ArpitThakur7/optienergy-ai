# OptiEnergy AI - presentation content and speaker notes

**10 slides · 9 minutes total.** The measurements below are from the actual demo run, not a proposed full experiment. Use the named exported graphs on the matching slides.

## 1. Title and research question (0:00-0:40)

**On slide:** OptiEnergy AI. Which optimization methods reach useful next-hour forecasts efficiently and stably?

**Speaker notes:** Electricity forecasting is the application, while optimization is the main research subject. I compare convex linear/ridge models and a non-convex ANN, holding conditions fixed within each family. I measure accuracy together with computation and stability.

## 2. Dataset and units (0:40-1:30)

**On slide:** UCI 235; 2,075,259 minute records; kWh = sum(kW)/60; 34,085 complete observed hours. Visual: `figures/dataset_eda.png`.

**Speaker notes:** Power and energy have different units. Only complete 60-minute hours form targets. At origin t, the previous hour is finished; I predict the next interval. Missing targets are never filled. Credit Hebrail and Berard, DOI:10.24432/C58K54, CC BY 4.0.

## 3. Leakage-safe features and splits (1:30-2:20)

**On slide:** 26 causal inputs; lags 1/3/6/12/24; rolling mean/std; cyclical calendar; chronological 70/15/15; boundary purge; train-only scaling.

**Speaker notes:** Forward fill is limited to six input hours. Every feature uses information available at the forecast origin. I purge labels crossing split boundaries, fit transformations on training rows, and test causality by changing future observations. The demo has 1656 train, 354 validation, and 356 test rows.

## 4. Objective and geometry (2:20-3:20)

**On slide:** J=||Aw-u||²/n + alpha*wᵀDw/2. Gradient=2Aᵀ(Aw-u)/n+alpha*Dw. Hessian=2AᵀA/n+alpha*D. GD: 0<eta<2/L.

**Speaker notes:** The Hessian is positive semidefinite, proving convexity. Redundant inputs make it singular; the numerical rank is 26/27. Ridge adds curvature without penalizing the bias. ANN parameters interact nonlinearly, so the global convex guarantee no longer applies. My gradient-check error is 1.15e-10.

## 5. Eight optimization methods (3:20-4:15)

**On slide:** Batch GD, SGD, mini-batch GD, Momentum, Nesterov, RMSprop, Adam, L-BFGS. Show two update equations plus state/cost table.

**Speaker notes:** The seven first-order variants are hand-written for linear models. Momentum stores direction, RMSprop stores squared gradients, and Adam stores first and second moments with bias correction. L-BFGS stores curvature pairs and uses a line search. I count every closure evaluation, because one accepted step may cost several gradients.

## 6. Fair experimental protocol (4:15-5:10)

**On slide:** Seeds 7/21/42; same initialization and split; equal example-processing budgets; separate updates/gradients/time; validation-only tuning; final test.

**Speaker notes:** An epoch is one data pass but has different update counts. Monitoring is separately counted, and timing includes it. One-factor sensitivity changes learning rate, batch size, momentum, regularization, schedule, epochs, or scaling. Demo sweeps representative pairings; full mode expands coverage. The direct sklearn solution is a reference rather than an iterative matched-budget method.

## 7. Measured results (5:10-6:15)

**On slide:** 126 runs; 6 diverged. Validation selection: **linear / Momentum**. Test RMSE **0.690 ± 0.012 kWh**; persistence **0.789 kWh**. Visual: `figures/leaderboard.png`.

**Speaker notes:** These are actual measurements on 2400 clock hours, not all four years for demo mode. The lowest validation mean selects the champion; a better observed test score does not change selection. Close top scores do not establish significance. The deployable model uses fixed seed 7, whose score is 0.692 kWh. Selection: {'model': 'linear', 'optimizer': 'Momentum', 'lr': np.float64(0.01), 'batch_size': np.int64(64), 'epochs': np.int64(10), 'momentum': np.float64(0.9), 'alpha': np.float64(0.0), 'scaling': 'standard', 'schedule': 'constant'}.

## 8. Convergence and sensitivity (6:15-7:15)

**On slide:** `figures/convergence_effective_passes.png` and `figures/sensitivity.png`.

**Speaker notes:** Read loss against data passes and time, not just epochs. The linear SGD baseline at .01 diverged; its failure remains in the logs. A full-data curvature bound does not guarantee per-sample stability. High momentum and learning rates can overshoot; small steps can underfit a short budget. State which comparison is baseline and which is tuned.

## 9. Dashboard demonstration (7:15-8:15)

**On slide:** Seven pages; interactive training; 2D/3D landscapes; historical energy prediction; exports.

**Demo sequence:** Open Optimizer Comparison, change the budget axis; open Playground, compare ill-conditioned quadratic and Rosenbrock; open Energy Prediction, choose a historical origin and show a 24-hour recursive forecast.

**Speaker notes:** Benchmark gradients isolate geometry. Recursive forecasts truncate observations at the origin and feed only predictions forward. Actual future readings appear solely for comparison. Multi-horizon accuracy has not been separately validated.

## 10. Conclusions and limitations (8:15-9:00)

**On slide:** Geometry and budgets matter. Reproducible code and real results. Limits: one household, short demo, three seeds. Next: full suite, rolling-origin validation, more households.

**Speaker notes:** This project demonstrates mathematical formulation, self-implemented methods, controlled experiments, and honest reporting. It does not declare a universally best optimizer. The next scientific step is stronger temporal evaluation and equal tuning resources across families. Questions are welcome.

## Before presenting

Run the dashboard and cache the pages. Keep the exported PNGs available offline. Rehearse the nine-minute script. Be ready to derive the gradient, explain kW-to-kWh conversion, and distinguish gradient calls from processed examples. The complete algorithm equations and 40 viva answers are in the companion notes.
