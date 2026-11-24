# Verification record

The notebook was executed from top to bottom in a fresh local Jupyter kernel using Windows 11, Python 3.13.2, and the exact dependency versions in `requirements-tested.txt`. All numerical computation used the CPU. The separate reduced CI profile was also executed in an isolated kernel and workspace.

## Executed checks

* All **30 Python cells** executed successfully, with no error outputs or warning messages in the final notebook.
* **40 core correctness tests** passed: gradients, Hessian action, optimizer updates, convex convergence, benchmark gradients, units, target handling, temporal separation, model persistence, reproducibility, divergence recovery, and all model/optimizer training pairings.
* **12 additional scientific tests** passed: next-hour seasonal alignment, forward-only baseline information, calendar embargoes, distinct inner/outer windows, train-only transforms, direct ridge objective mapping, all reference model families, paired bootstrap correctness/reproducibility, invalid alignment rejection, incomplete-seed exclusion, and recursive causality.
* **9 repository/evidence checks** passed. They validate notebook schema and compilation, standalone imports, absence of opaque source/data blobs, error-free saved outputs, support files, the committed portfolio profile, score recomputation from predictions, and agreement with frozen selection.
* The optimization lab contains **126 attempted measured runs** across 42 configurations/references and three seeds. Its cache-hit rows retain original measured training times. Divergence is retained and excluded from final configuration selection.
* The later-period forecasting study executed **69 candidate-fold-seed scores**, **13 final evaluations**, and **9 feature-group ablations**. It used three calendar validation windows and a separate later-period holdout of **6,296** common-support observations.
* The held-out result tables were independently recomputed from the committed prediction CSV. Final metrics agree to at least ten decimal places.
* The notebook contains **12 rendered static figures**. The main exported figures were visually inspected for readable labels, correctly separated validation/test meanings, and layout. The plotting code preserves the notebook's inline backend.
* The CI execution profile completed the full notebook workflow with reduced training/evaluation budgets. Its outputs are stored separately and do not overwrite the committed research evidence.

## Scope of the claims

The forecasting benchmark uses the available multi-year chronology, while the controlled optimizer lab is an early-window demonstration. The expanded multi-year optimizer grid is implemented but has not been executed. These are different studies.

The GitHub Actions workflow is included and its execution commands were tested locally. It has not been run on a hosted GitHub repository in this delivery. A hosted Google Colab session has not been executed; the notebook uses portable Python cells and conditional Colab rendering configuration.

The approximate block-bootstrap interval applies to fixed-seed error differences on the one held-out household period. It is not an individual forecast interval, does not include training-sample uncertainty, and does not establish external-household performance.
