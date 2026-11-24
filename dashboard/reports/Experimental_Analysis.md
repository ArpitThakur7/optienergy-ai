# Measured experimental analysis

This is the **demo** study: 2400 hourly records before feature filtering, baseline 10 epochs, seeds [7, 21, 42]. 126 runs were attempted; 6 diverged. Cached repetitions retain original measured timings. Dataset raw SHA-256: `4259c9d7ece5dbee9ab8d53682baac68d791c864f0f64a52b4043cb3b90894b7`.

Chronological usable row counts: 1656 train, 354 validation, 356 test. Test origins: 2007-03-11 20:00:00 through 2007-03-26 16:00:00.

The champion is **linear / Momentum**, chosen by mean validation RMSE **0.5940 kWh**. Final test RMSE is **0.6904 ± 0.0117 kWh**, sample SD. The fixed deployment seed 7 has RMSE 0.6921 kWh, which is not the family mean. Persistence test RMSE is 0.7886 kWh.

The leading validation scores are close. Their order does not demonstrate statistical significance; another row can have better test RMSE without becoming the selected champion. Test observations never guide selection.

## Validation-selected leaderboard

| Model | Optimizer | Val RMSE | Test RMSE ± SD | Test MAE | Test R² | Seconds ± SD |
|---|---|---:|---:|---:|---:|---:|
| linear | Momentum | 0.5940 | 0.6904 ± 0.0117 | 0.5109 | 0.5693 | 0.0060 ± 0.0002 |
| linear | Adam | 0.5943 | 0.6791 ± 0.0087 | 0.4899 | 0.5834 | 0.0155 ± 0.0009 |
| ridge | Momentum | 0.5948 | 0.6906 ± 0.0115 | 0.5121 | 0.5690 | 0.0066 ± 0.0005 |
| linear | Nesterov | 0.5980 | 0.6853 ± 0.0102 | 0.5103 | 0.5756 | 0.0073 ± 0.0002 |
| ridge | Nesterov | 0.5989 | 0.6856 ± 0.0101 | 0.5113 | 0.5754 | 0.0071 ± 0.0004 |
| ridge | Adam | 0.6021 | 0.6841 ± 0.0074 | 0.4970 | 0.5771 | 0.0093 ± 0.0001 |
| linear | L-BFGS | 0.6032 | 0.6869 ± 0.0000 | 0.5072 | 0.5738 | 0.0024 ± 0.0007 |
| ridge | L-BFGS | 0.6038 | 0.6861 ± 0.0000 | 0.5070 | 0.5747 | 0.0028 ± 0.0002 |
| ridge | RMSprop | 0.6051 | 0.6790 ± 0.0177 | 0.4968 | 0.5833 | 0.0089 ± 0.0013 |
| linear | RMSprop | 0.6052 | 0.6793 ± 0.0181 | 0.4970 | 0.5830 | 0.0077 ± 0.0001 |
| ann | SGD | 0.6130 | 0.7119 ± 0.0327 | 0.5031 | 0.5416 | 8.6610 ± 0.2001 |
| ann | Momentum | 0.6221 | 0.7144 ± 0.0312 | 0.5307 | 0.5384 | 0.1719 ± 0.0019 |
| ann | Nesterov | 0.6275 | 0.7123 ± 0.0256 | 0.5308 | 0.5413 | 0.1663 ± 0.0028 |
| sklearn | Direct solve | 0.6283 | 0.6746 ± 0.0000 | 0.5025 | 0.5889 | 0.0019 ± 0.0004 |
| linear | Mini-batch GD | 0.6359 | 0.6889 ± 0.0050 | 0.5143 | 0.5713 | 0.0063 ± 0.0003 |
| ridge | Mini-batch GD | 0.6363 | 0.6891 ± 0.0050 | 0.5145 | 0.5710 | 0.0055 ± 0.0001 |
| ann | Adam | 0.6445 | 0.7070 ± 0.0155 | 0.5111 | 0.5483 | 0.2142 ± 0.0082 |
| ann | RMSprop | 0.6572 | 0.7183 ± 0.0289 | 0.5227 | 0.5333 | 0.1890 ± 0.0096 |
| ann | L-BFGS | 0.7112 | 0.7820 ± 0.0689 | 0.5940 | 0.4447 | 0.0167 ± 0.0007 |
| ann | Mini-batch GD | 0.7195 | 0.7674 ± 0.0509 | 0.5952 | 0.4665 | 0.1591 ± 0.0034 |
| linear | Batch GD | 0.8122 | 0.8518 ± 0.0000 | 0.6898 | 0.3445 | 0.0023 ± 0.0001 |
| ridge | Batch GD | 0.8123 | 0.8519 ± 0.0000 | 0.6898 | 0.3444 | 0.0020 ± 0.0000 |
| ann | Batch GD | 1.0553 | 1.0277 ± 0.0148 | 0.8441 | 0.0458 | 0.6214 ± 1.0295 |

## What the measured study supports

* Baseline divergence: linear/SGD: 3 failed seeds; ridge/SGD: 3 failed seeds. Failed runs retain finite checkpoints for diagnostics but are ineligible for final selection. The full suite includes smaller learning-rate sweeps for all first-order pairings.
* The unregularized standardized Hessian has numerical rank 26 of 27. Its full condition number is undefined at tolerance; positive-spectrum condition number is 325.93.
* Ridge alpha=.01 raises the smallest eigenvalue to 0.01000. Its finite full condition number 1251.48 is not directly comparable to the unregularized positive-spectrum-only number.
* The full-data GD step bound is 2/L=0.15994. SGD divergence below that bound is possible because individual samples have higher curvature.
* The analytic gradient's central-difference relative error is 1.149e-10.
* Full-batch linear methods are deterministic across seeds; zero predictive SD is expected. ANN initialization and mini-batch order produce variability.

## One-factor sensitivity

### Learning rate: linear / Adam

| lr | Validation RMSE mean | Sample SD | Valid seeds / attempts |
|---|---:|---:|---:|
| 0.0001 | 1.0119 | 0.0024 | 3 / 3 |
| 0.001 | 0.7169 | 0.0006 | 3 / 3 |
| 0.01 | 0.6021 | 0.0055 | 3 / 3 |
| 0.1 | 0.6417 | 0.0475 | 3 / 3 |

### Batch size: linear / Adam

| batch_size | Validation RMSE mean | Sample SD | Valid seeds / attempts |
|---|---:|---:|---:|
| 16.0 | 0.5947 | 0.0050 | 3 / 3 |
| 32.0 | 0.6029 | 0.0085 | 3 / 3 |
| 64.0 | 0.6021 | 0.0055 | 3 / 3 |
| 128.0 | 0.6337 | 0.0109 | 3 / 3 |

### Momentum: linear / Momentum

| momentum | Validation RMSE mean | Sample SD | Valid seeds / attempts |
|---|---:|---:|---:|
| 0.5 | 0.6199 | 0.0092 | 3 / 3 |
| 0.9 | 0.5940 | 0.0011 | 3 / 3 |
| 0.99 | 0.9121 | 0.1241 | 3 / 3 |

### Regularization: ridge / Adam

| alpha | Validation RMSE mean | Sample SD | Valid seeds / attempts |
|---|---:|---:|---:|
| 0.0 | 0.6021 | 0.0055 | 3 / 3 |
| 0.001 | 0.6022 | 0.0055 | 3 / 3 |
| 0.01 | 0.6024 | 0.0056 | 3 / 3 |
| 0.1 | 0.6065 | 0.0062 | 3 / 3 |

### Scaling: linear / Adam

| scaling | Validation RMSE mean | Sample SD | Valid seeds / attempts |
|---|---:|---:|---:|
| minmax | 0.7619 | 0.0161 | 3 / 3 |
| none | 0.6689 | 0.0190 | 3 / 3 |
| robust | 0.6466 | 0.0093 | 3 / 3 |
| standard | 0.6021 | 0.0055 | 3 / 3 |

### Schedule: linear / Adam

| schedule | Validation RMSE mean | Sample SD | Valid seeds / attempts |
|---|---:|---:|---:|
| constant | 0.6021 | 0.0055 | 3 / 3 |
| decay | 0.6097 | 0.0111 | 3 / 3 |

### Epochs: linear / Adam

| epochs | Validation RMSE mean | Sample SD | Valid seeds / attempts |
|---|---:|---:|---:|
| 5.0 | 0.6229 | 0.0053 | 3 / 3 |
| 10.0 | 0.6021 | 0.0055 | 3 / 3 |
| 20.0 | 0.5943 | 0.0106 | 3 / 3 |

## Reading the graphs and counters

The fixed-baseline convergence graphs hold model, objective, initial weights, split, and baseline hyperparameters constant. The tuned leaderboard can vary one selected setting per pairing. These answer different questions; demo tuning coverage differs across pairings. Effective data passes, gradient calls, optimizer updates, and seconds must be interpreted together. Monitoring overhead is separately counted and included in seconds. Checkpoint stability and threshold time depend on monitoring frequency. Memory is checkpoint-sampled whole-process RSS.

A lower final training objective can generalize worse; returned models use validation-best checkpoints. `final_training_loss` and `final_gradient_norm` refer to the last finite checkpoint, not necessarily the restored model. `best_epoch` identifies the returned checkpoint. Missing threshold time means not reached. Failed runs remain in `runs.csv` and `baseline_summary.csv`; absence from the final leaderboard is deliberate.

## Limits

This study does not establish multi-household generalization or universal optimizer superiority. Three-seed SD is not a confidence interval. The supplied demo does not measure full-year performance; full mode must be executed separately for the complete history and sensitivity coverage. Recursive 24-hour forecasts are a causal extension with no separately claimed multi-horizon score. Reference data: Hebrail & Berard, UCI, DOI:10.24432/C58K54.
