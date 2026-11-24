# Viva questions and concise answers

1. **What is being optimized?** Standardized next-hour prediction MSE, optionally plus alpha/2 times squared non-bias weights. This is distinct from the physical energy target in kWh.
2. **Why is hourly power not already energy?** kW is power. Each minute contributes power/60 kWh; summing 60 valid readings integrates over one hour.
3. **What exactly is the forecast origin?** Index t means the previous hour has finished. We estimate energy from t to t+1 hour with no measurements from that interval.
4. **Why avoid random train/test splits?** Nearby time-series rows share lag windows and future regimes. Random splitting would violate the intended chronological forecast setting.
5. **Why purge split boundaries?** A row just before a validation origin can have its label at that origin; purging keeps training labels strictly before validation origins.
6. **How do you prove there is no future leakage?** Features use causal shifts/rolling windows, inputs fill only forward, transformations fit on train, history truncates before recursive features, and tests mutate future observations without changing earlier features or forecasts.
7. **Why not interpolate all gaps?** Two-sided interpolation uses future endpoints. We only forward-fill short input gaps and never fill targets.
8. **Is linear regression always strictly convex?** It is convex; strict convexity requires positive-definite X-transpose-X (after intercept augmentation). Redundant features can make it singular.
9. **Why does ridge help optimization?** It adds curvature to penalized directions and improves conditioning while trading prediction bias for lower variance. The intercept remains unpenalized.
10. **What is the gradient?** For augmented design A, 2 A-transpose(Aw-y)/n + alpha D w, where D has a zero at the intercept position.
11. **What is the Hessian?** 2 A-transpose-A/n + alpha D, constant for linear squared loss. Its eigenvalues determine curvature and stable full-batch steps.
12. **What is a safe batch-GD step?** For this quadratic, 0<eta<2/L, where L is the maximum Hessian eigenvalue; eta<=1/L guarantees descent.
13. **Does that bound guarantee SGD stability?** No. Single-sample Hessians may have much larger eigenvalues than the mean Hessian, especially with outliers.
14. **Why scale features?** Unequal feature scales elongate contours and constrain stable steps. Scaling changes the Hessian and often reduces directional imbalance.
15. **Why is the ANN non-convex?** The parameters interact through matrix products and nonlinear activations. A zero gradient need not identify a global minimum.
16. **Why are batch GD, SGD, and mini-batch GD different if the update equation matches?** Their gradient estimates use n, 1, or b samples. Noise, per-update cost, and the number of updates per data pass differ.
17. **What does momentum store?** A decaying accumulated update/gradient direction. It can accelerate consistent directions but can overshoot with high momentum.
18. **How is Nesterov different?** The NumPy implementation evaluates the gradient at a momentum lookahead point before completing the velocity update. PyTorch uses its documented buffer form.
19. **What does RMSprop adapt?** Per-coordinate step scales from an exponentially smoothed squared gradient. Its epsilon avoids division by zero.
20. **Why does Adam use bias correction?** Zero-initialized moment estimates underestimate early moments; dividing by 1-beta^t corrects that initialization bias.
21. **Why not assume Adam wins?** Adaptive steps can help, but results depend on geometry, hyperparameters, stochasticity, regularization, and budget. We rank measured validation outcomes.
22. **Why use L-BFGS?** It approximates inverse-Hessian actions from recent curvature pairs with limited memory and often makes useful full-data progress with relatively few accepted steps.
23. **Why count L-BFGS function evaluations?** A line search may evaluate several trial steps for one accepted update. Ignoring those calls overstates efficiency.
24. **Are epochs a fair budget?** Equal passes match data-gradient processing for first-order methods, but do not match update counts, overhead, or wall time. We display all of them.
25. **What is the sklearn baseline for?** It gives an unregularized direct least-squares reference. It is not an eighth custom iterative optimizer.
26. **How are hyperparameters selected?** Mean validation RMSE across seeds; test data is evaluated only for the selected configuration per model/optimizer. The champion's deployment seed is pre-fixed to 7.
27. **Why three seeds?** They expose some stochastic variability while keeping the demo tractable. Three are insufficient for strong statistical claims.
28. **Is sample SD a confidence interval?** No. It describes spread across the chosen seeds; data uncertainty and temporal regime uncertainty are different.
29. **Why can R-squared be negative?** The model can have larger squared error than predicting the test-set mean. Negative values are retained.
30. **Why keep diverged runs?** Divergence is a meaningful learning-rate/stability outcome. Failed runs must not be silently dropped from the experimental record or assigned invented scores.
31. **What is the difference between best and final loss?** The returned model is the lowest-validation-loss checkpoint. Final logs describe the last finite checkpoint, which may generalize less well.
32. **Does lower training loss guarantee lower test error?** No. Overfitting and temporal distribution shift can worsen generalization even while optimizing the training objective.
33. **How does recursive forecasting avoid leakage?** Truncate observed history at t, predict t+1, append that prediction, then build the next feature row. Actual future readings are for evaluation only.
34. **What is the weakness of recursive forecasting?** Prediction errors feed later inputs and compound. The one-step model was not trained specifically for multi-horizon accuracy.
35. **Why a persistence baseline?** Predicting next hour equals the last observed hour tests whether learned models add value beyond simple temporal continuity.
36. **What does the benchmark playground teach?** Conditioned versus ill-conditioned convex geometry, narrow valleys, multiple minima, learning-rate instability, and how optimizer trajectories react.
37. **How is custom differentiation validated?** Central finite differences against analytic gradients; Hessian-vector differences; known optimizer first steps; convergence on a convex quadratic.
38. **What is measured memory?** Checkpoint-sampled process RSS. It includes Python, libraries, and loaded models, so it is not isolated optimizer memory and may miss transient peaks.
39. **What is reproducible about timing?** Software versions, CPU metadata, threading, and seeds are logged. Wall time still depends on machine load, caching, and hardware; exact matching is not promised.
40. **What would you improve next?** Run the full suite, use rolling-origin validation, evaluate multiple households and seasons, add weather/occupancy covariates, and directly evaluate multi-horizon forecasting.
