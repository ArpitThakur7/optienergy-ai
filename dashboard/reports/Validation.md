# Verification record

The delivered demo was executed in Windows 11 with Python 3.13.2. The complete installed package versions appear in `requirements-tested.txt` and in the experiment manifest. Computation used the CPU; PyTorch was configured for one thread and deterministic algorithms.

## Executed verification

* **64 tests passed.** These cover gradients/Hessian action, optimizer equations and convergence, data units and missing values, temporal separation, train-only scaling, all model/optimizer pairings, checkpoint recovery, persistence, causal recursive forecasts, experiment-plan fairness, seven Streamlit pages, navigation, and a submitted training form.
* **126 freshly trained measured runs** in the final demo: 42 configurations including the reference, each with seeds 7, 21, and 42. Results cover the first 2,400 hourly records before feature filtering. Six failed SGD repetitions are retained and excluded from final selection.
* The deployable model and leaderboard are selected using validation results. Its final test RMSE is 0.690384 kWh, sample SD 0.011677 kWh across the three selected repetitions. This is the family mean; the fixed seed-7 inference artifact has a separate score.
* The numerical gradient check on real training rows produced relative error 1.149e-10.
* The PDF report contains **nine pages**, rendered to PNG using Poppler and visually inspected for layout, tables, figures, and clipping.
* The Colab notebook was schema-validated, and its project modules/commands were executed locally. The Google Colab upload/download flow itself has not been run in a hosted Colab session.
* The ZIP archive passes its CRC integrity check and contains the source, tests, cached hourly data, final measured results, models, notebook, and reports. Raw downloads, experiment caches, and bytecode are excluded.

## Limits of verification

The full multi-year sensitivity suite is implemented but has not been executed. The live in-app browser preview was unavailable because the automatic approval service timed out twice; the seven page-rendering checks and training-form test passed through Streamlit's AppTest framework. The dashboard was therefore programmatically tested, but its rendered browser appearance was not visually inspected.

The installed PyTorch build included CUDA support, but every model and tensor used CPU execution. Timing varies with hardware and process load. The supplied exact-version file records the tested versions, while the main requirements file permits compatible releases for local and Colab installations.
