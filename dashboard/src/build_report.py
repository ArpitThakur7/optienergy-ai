"""Generate a nine-page academic PDF, measured analysis, and a nine-minute presentation."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                               Spacer, Table, TableStyle)

from config import RESULTS, ROOT


def one_factor(runs, model, optimizer, parameter):
    frame = runs[(runs.model == model) & (runs.optimizer == optimizer)].copy()
    base = frame[frame.sweep == "baseline"].iloc[0]
    for field in ("lr", "batch_size", "momentum", "alpha", "epochs", "scaling", "schedule"):
        if field != parameter:
            frame = frame[frame[field] == base[field]]
    frame["eligible_val_rmse"] = frame.val_rmse.where(frame.status != "diverged")
    return frame.groupby(parameter).agg(mean=("eligible_val_rmse", "mean"), sd=("eligible_val_rmse", "std"),
                                        valid_seeds=("eligible_val_rmse", "count"), attempts=("seed", "size")).reset_index()


def build_report(mode="demo"):
    directory = RESULTS / mode
    report_dir = ROOT / "reports" if mode == "demo" else ROOT / "reports/full"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    analysis = json.loads((directory / "analysis.json").read_text())
    board, runs = pd.read_csv(directory / "leaderboard.csv"), pd.read_csv(directory / "runs.csv")
    history = pd.read_csv(directory / "histories.csv")
    champion = board.iloc[0]
    chosen_cfg = runs[runs.config_id == analysis["champion"]["config_id"]].iloc[0]
    seed_prediction = pd.read_csv(directory / "predictions.csv")
    seed_rmse = np.sqrt(np.mean((seed_prediction.actual_kwh-seed_prediction.predicted_kwh)**2))
    h = analysis["hessian"]
    failure_groups = runs[(runs.sweep == "baseline") & (runs.status == "diverged")].groupby(["model", "optimizer"]).size()
    failure_text = "; ".join(f"{model} / {optimizer}: {count} failed seeds" for (model, optimizer), count in failure_groups.items()) or "No baseline divergence was observed"
    ntrain, nval, ntest = [manifest["split"][k]["rows"] for k in ("train", "validation", "test")]
    scope = (f"First {manifest['max_hours']:,} clock hours" if manifest["max_hours"] else "Full available hourly history")
    sensitivities = [("Learning rate", "linear", "Adam", "lr"), ("Batch size", "linear", "Adam", "batch_size"),
                     ("Momentum", "linear", "Momentum", "momentum"), ("Scaling", "linear", "Adam", "scaling")]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(8.2, 5.3), layout="constrained")
    for ax, (label, model, optimizer, parameter) in zip(axs.flat, sensitivities):
        group = one_factor(runs, model, optimizer, parameter)
        ax.errorbar(np.arange(len(group)), group["mean"], yerr=group.sd.fillna(0), marker="o", capsize=4)
        ax.set_xticks(np.arange(len(group)), group[parameter].astype(str), rotation=15)
        ax.set(xlabel=label, ylabel="Validation RMSE (kWh)", title=f"{model} / {optimizer}")
        ax.grid(alpha=.2)
    fig.savefig(directory / "figures/sensitivity.png", dpi=180)
    plt.close(fig)
    # Full-history EDA is descriptive; it does not choose features or tune parameters.
    from src.data_loader import load_hourly
    hourly, _ = load_hourly()
    fig, axs = plt.subplots(2, 1, figsize=(8.2, 4.5), layout="constrained")
    daily = hourly.energy_kwh.resample("D").sum(min_count=24)
    axs[0].plot(daily.index, daily, lw=.7)
    axs[0].set(ylabel="Daily energy (kWh)", title="Observed complete days: one household")
    pattern = hourly.energy_kwh.groupby((hourly.index.hour-1) % 24).mean()
    axs[1].bar(pattern.index, pattern)
    axs[1].set(xlabel="Hour at interval start (local clock)", ylabel="Mean hourly kWh")
    fig.savefig(directory / "figures/dataset_eda.png", dpi=170)
    plt.close(fig)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=30, leading=33,
                              textColor=colors.HexColor("#112740"), spaceAfter=15))
    styles.add(ParagraphStyle(name="Deck", fontSize=13, leading=19, textColor=colors.HexColor("#42617b"), spaceAfter=14))
    styles.add(ParagraphStyle(name="BodyReport", fontSize=9.3, leading=13.5, spaceAfter=8))
    styles.add(ParagraphStyle(name="SmallReport", fontSize=8, leading=11, spaceAfter=6))
    styles.add(ParagraphStyle(name="EqReport", fontName="Courier", fontSize=9, leading=13,
                              backColor=colors.HexColor("#edf5f7"), borderPadding=8, spaceBefore=5, spaceAfter=12))
    styles["Heading1"].textColor = colors.HexColor("#112740")
    styles["Heading2"].fontSize, styles["Heading2"].leading = 12, 15
    story = []

    def p(text, style="BodyReport"):
        story.append(Paragraph(text, styles[style]))

    def title(number, text):
        p(f"{number:02d} / OPTIMIZATION IN MACHINE LEARNING", "SmallReport")
        p(text, "Heading1")

    def sub(text):
        p(text, "Heading2")

    def table(headers, rows, widths=None, font=8):
        data = [[Paragraph(html.escape(str(v)), styles["SmallReport"]) for v in headers]]
        data += [[Paragraph(html.escape(str(v)), styles["SmallReport"]) for v in row] for row in rows]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dceaf1")),
                               ("VALIGN", (0,0), (-1,-1), "TOP"),
                               ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f8fb")]),
                               ("BOTTOMPADDING", (0,0), (-1,-1), 3), ("TOPPADDING", (0,0), (-1,-1), 3),
                               ("LINEBELOW", (0,0), (-1,0), .6, colors.HexColor("#b6cbd9"))]))
        story.append(t)
        story.append(Spacer(1, 8))

    def image(name, height=None):
        path = directory / "figures" / name
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            w, hh = im.size
        width = 475
        target_h = height or width * hh / w
        story.append(Image(str(path), width=width, height=target_h))
        story.append(Spacer(1, 5))

    def page():
        story.append(PageBreak())

    p("B.TECH CSE / RESEARCH PROJECT", "SmallReport")
    p("OptiEnergy AI", "ReportTitle")
    p("A Comparative Study of Optimization Algorithms for Smart Energy Consumption Prediction", "Deck")
    p("Course: Optimization in Machine Learning | Reproducible CPU study | 24 November 2026", "SmallReport")
    sub("Abstract")
    p("This project investigates how optimization algorithms affect household next-hour energy prediction. "
      "A causal pipeline converts minute-level UCI power observations into hourly energy and creates lagged, rolling, and cyclical calendar features. "
      "Unregularized linear regression provides a convex setting, ridge adds controlled curvature, and a fixed small Tanh neural network provides a non-convex setting. "
      "Seven first-order update variants and L-BFGS are instrumented for data passes, updates, gradient/function evaluations, elapsed time, gradient norm, and checkpoint stability.")
    p(f"The included {mode} study executed {analysis['run_count']} runs across seeds 7, 21, and 42. "
      f"The validation-selected family was <b>{champion['model']} / {champion.optimizer}</b>, with mean validation RMSE "
      f"{champion.val_rmse_mean:.4f} kWh. Its final test RMSE was {champion.test_rmse_mean:.4f} +/- {champion.test_rmse_std:.4f} kWh "
      f"(sample SD), compared with persistence RMSE {analysis['persistence_test']['rmse']:.4f} kWh. "
      f"There were {analysis['diverged_runs']} diverged runs, retained as stability outcomes. "
      "These measurements do not establish a universally best optimizer.")
    sub("Research questions")
    p("1. How do curvature, conditioning, and scaling affect stable learning rates and convergence?<br/>"
      "2. How do momentum and adaptive updates trade computational cost against prediction error?<br/>"
      "3. How sensitive are results to learning rate, batch size, regularization, schedule, and seed?")
    sub("Measured scope")
    table(["Setting", "Included measurement"], [["Data budget", scope],
        ["Baseline / repetitions", f"{manifest['epochs']} epochs; 3 seeds"],
        ["Chronological train / validation / test", f"{ntrain:,} / {nval:,} / {ntest:,} usable rows"],
        ["Full multi-year sensitivity suite", "Implemented; not executed in this demo report" if mode == "demo" else "Executed for this full report"]], [200, 275])
    p("Keywords: gradient descent; stochastic optimization; conditioning; causal forecasting; reproducible experiments.", "SmallReport")
    page()

    title(2, "Dataset and causal forecasting pipeline")
    p("The UCI Individual Household Electric Power Consumption dataset contains one-minute measurements from a single household in Sceaux, France. "
      "It is distributed under CC BY 4.0 (Hebrail &amp; Berard; DOI:10.24432/C58K54). The official source was downloaded and its text-file SHA-256 stored in the run manifest.")
    table(["Quantity", "Observed value"], [["Raw minute rows", f"{manifest['dataset']['raw_rows']:,}"],
       ["Missing or invalid minutes", f"{manifest['dataset']['missing_or_invalid_minutes']:,}"],
       ["Complete hourly energy labels", f"{manifest['dataset']['complete_hours']:,}"],
       ["Incomplete / partial hours", f"{manifest['dataset']['incomplete_hours']:,}"]], [260, 215])
    p("energy_kWh(t) = sum of minute-average power_kW / 60", "EqReport")
    p("Hourly labels require 60 valid nonnegative readings. The index is the right endpoint: a row at 14:00 represents [13:00,14:00). "
      "At origin t, the target is the next interval [t,t+1h), ending at t+1h. Input gaps alone may forward-fill for at most six hours. "
      "No backward filling, centered rolling averages, or target imputation is used.")
    sub("Features and temporal separation")
    p("The 26 inputs are current energy, a missing-input indicator, lags 1/3/6/12/24h, rolling mean and standard deviation for 3/6/12/24h, "
      "hour/day/weekday/month/weekend, and sine/cosine hour/weekday/month encodings. Rolling features include the just-finished hour, which is available at the origin. "
      "The 70/15/15 split is chronological; labels reaching the next partition are purged. Scaling and target standardization use only training rows.")
    image("dataset_eda.png", 251)
    p("Figure 1. Full-history descriptive EDA; model selection uses only the stated train/validation periods. Daily totals require complete observed days.", "SmallReport")
    page()

    title(3, "Objective, curvature, and convergence theory")
    p("Let A contain the transformed features and an appended intercept column; u is the training-standardized target. "
      "Let D be diagonal with ones for weights and zero for the intercept.")
    p("J(w) = ||Aw - u||^2 / n + (alpha/2) w^T D w<br/>"
      "grad J = 2 A^T(Aw-u)/n + alpha D w<br/>H = 2 A^T A/n + alpha D", "EqReport")
    p("For any v, v-transpose H v is nonnegative, so the linear objective is convex. A positive-definite H gives a unique minimizer; "
      "redundant or constant features can make unregularized regression singular. Ridge penalizes every weight but not the intercept, increasing curvature in otherwise flat directions. "
      "Its effect is not a uniform eigenvalue shift because the intercept is unpenalized.")
    sub("Measured geometry")
    table(["Train-standardized linear design", "Value"], [
        ["Hessian dimension / numerical rank", f"{h['dimension']} / {h['numerical_rank']}"],
        ["Largest eigenvalue L", f"{h['max_eigenvalue']:.6f}"],
        ["Full condition number", "Undefined / singular at numerical tolerance" if h['condition_number'] is None else f"{h['condition_number']:.3f}"],
        ["Positive-spectrum condition number", f"{h['positive_spectrum_condition_number']:.3f}"],
        ["Batch-GD stable upper step 2/L", f"{h['safe_gd_lr_upper_bound']:.6f}"],
        ["Ridge alpha=.01 minimum eigenvalue", f"{h['ridge_alpha_0.01_min_eigenvalue']:.6f}"],
        ["Ridge alpha=.01 condition number", f"{h['ridge_alpha_0.01_condition_number']:.3f}"],
        ["Central-difference relative gradient error", f"{analysis['gradient_check_relative_error']:.3e}"]], [300, 175])
    p("For a quadratic, GD's error along an eigenvector contracts by 1-eta*lambda. Steps 0&lt;eta&lt;2/L are stable; eta&lt;=1/L ensures descent. "
      "Small curvature eigenvalues slow progress. The full-data bound does not bound each sample's curvature and therefore does not certify SGD stability. "
      "Feature scaling alters the geometry; it is evaluated as a separate experimental factor.")
    sub("Non-convex setting")
    p("The ANN has widths 26-32-16-1 with Tanh activations and the same MSE/L2 convention. Parameter interactions create a non-convex landscape. "
      "A small gradient can indicate a local minimum, a saddle, or a saturated region; it does not certify global optimality. "
      "The comparison fixes architecture and seed-specific initial weights within this family.")
    p("The numerical rank threshold is L times 1e-10. Roundoff-scale negative eigenvalues would not establish non-convexity. "
      "The finite positive-spectrum condition number is not the condition number of the singular full Hessian.", "SmallReport")
    page()

    title(4, "Optimization methods and implementation")
    p("Let g be the mean batch gradient including the exact L2 gradient. State begins at zero. First-order methods are written in NumPy for linear/ridge models; "
      "the ANN uses corresponding PyTorch optimizers. Epsilon is 1e-8.")
    table(["Method", "Implemented principle", "Cost / extra state"], [
        ["Batch GD", "w <- w - eta*g; all n rows", "O(np) / O(p)"],
        ["SGD", "Same update; one sample per step", "O(p) / O(p)"],
        ["Mini-batch GD", "Same update; b samples per step", "O(bp) / O(p)"],
        ["Momentum", "v <- beta*v-eta*g; w <- w+v", "O(bp) / O(p)"],
        ["Nesterov", "NumPy gradient at w+beta*v, then velocity update", "O(bp) / O(p)"],
        ["RMSprop", "s <- .99*s+.01*g^2; divide g by sqrt(s)+eps", "O(bp) / O(p)"],
        ["Adam", "First/second moments .9/.999, bias correction, coordinate scaling", "O(bp) / O(p)"],
        ["L-BFGS", "Inverse-Hessian action from 10 curvature pairs; line search", "O(qnp+mp) / O(mp)"]], [85, 265, 125])
    p("For L-BFGS, m=10 history pairs and q is the number of full-data evaluations for an accepted update. For neural models, replace the linear bp term "
      "by forward/backward network cost. Python/Autograd overhead makes sample-wise ANN SGD substantially different from vectorized linear SGD.")
    sub("Implementation details that matter")
    p("Adam uses per-update bias correction m/(1-.9^t), s/(1-.999^t); RMSprop has no such correction. Biases are not L2-penalized. "
      "The NumPy Nesterov code uses an explicit lookahead; PyTorch uses its standard momentum-buffer parameterization. This convention difference is disclosed, "
      "and conclusions compare optimizers within a model family rather than assuming identical iterates across libraries.")
    p("SciPy uses unconstrained L-BFGS-B for linear models; PyTorch uses strong-Wolfe L-BFGS for the ANN. Both are given a hard full-data closure budget. "
      "Accepted steps and closure evaluations are recorded separately. Learning rate, batch size, and momentum sweeps do not apply to these line-search runs.")
    sub("Validation of the implementation")
    p("Tests check analytic gradients against central differences, Hessian-vector differences, intercept regularization, exact first optimizer steps, "
      "convex benchmark convergence, all model/optimizer training combinations, repeated-seed determinism, and model save/load equivalence. "
      "Causality tests mutate future values and verify that earlier features and recursive predictions remain unchanged.")
    sub("Reference solver")
    p("Scikit-learn LinearRegression supplies a direct least-squares reference. It shares feature/target transformations and temporal partitions, "
      "but is not an iterative budget-matched optimizer. Its result is not described as an Adam result merely because other runs use Adam.")
    page()

    title(5, "Controlled experimental design")
    p(f"Measured scope: {scope.lower()}. Training has {ntrain} rows; validation {nval}; test {ntest}. "
      f"Baseline runs use {manifest['epochs']} epochs, eta=.01, batch size 64 for mini-batch methods, momentum=.9, and standard scaling. "
      "Linear and ANN use alpha=0; ridge uses alpha=.01. Linear initialization is zero; neural initialization is paired by seed.")
    table(["Experimental variable", "Values / protocol"], [
        ["Seeds", "7, 21, 42; mean and sample SD"],
        ["Learning rate", "0.0001, 0.001, 0.01, 0.1"], ["Batch size", "16, 32, 64, 128"],
        ["Momentum", "0.5, 0.9, 0.99"], ["Regularization alpha", "0, 0.001, 0.01, 0.1"],
        ["Schedule", "Constant; eta0/(1+.05*(epoch-1))"],
        ["Epoch sensitivity", "5/10/20 in demo; 10/30/60 in full mode"],
        ["Scaling", "Standard, min-max, robust, none"],
        ["Selection", "One factor at a time; mean validation RMSE; all 3 runs finite"],
        ["Final test", "Selected configuration per model/optimizer; no test-driven tuning"]], [160, 315])
    sub("Budget accounting")
    p("The first-order primary budget is epochs times n processed examples. The update count differs: one per epoch for batch GD, n per epoch for SGD, "
      "and ceil(n/b) per epoch for mini-batch methods. L-BFGS cannot exceed the same example-gradient budget, but may stop sooner due to convergence or line-search behavior. "
      "Monitoring full gradients and validation forwards is separately counted and included in measured elapsed time.")
    selected = runs[(runs.sweep == "baseline") & (runs.model == "linear") & (runs.seed == 7)]
    table(["Linear baseline / seed 7", "Updates", "Grad calls", "Data passes", "Status"], [
        [r.optimizer, int(r.updates), int(r.gradient_evaluations), f"{r.effective_passes:.2f}", r.status]
        for _, r in selected.iterrows()], [135, 60, 70, 70, 140])
    p("The demo sweeps representative pairings rather than every combination. Its tuned leaderboard therefore has unequal search budgets. "
      "Baseline curves hold settings fixed and support the cleanest within-family optimizer comparison. Full mode expands each applicable one-factor study. "
      "No full-suite results are claimed by this demo report.", "SmallReport")
    page()

    title(6, "Final predictive results")
    p("Rows are ordered by mean validation RMSE. Test metrics are revealed after configuration selection; lower test error is not used to reorder or choose the champion. "
      "Errors are in kWh and +/- is sample SD over three seeds. All unsuccessful baseline attempts remain in runs.csv and baseline_summary.csv.")
    table(["Model / optimizer", "Val RMSE", "Test RMSE +/- SD", "Test MAE", "Test R2"], [
        [f"{r['model']} / {r.optimizer}", f"{r.val_rmse_mean:.3f}", f"{r.test_rmse_mean:.3f} +/- {r.test_rmse_std:.3f}",
         f"{r.test_mae_mean:.3f}", f"{r.test_r2_mean:.3f}"] for _, r in board.iterrows()], [155, 65, 115, 70, 70])
    p(f"The selected {champion['model']} / {champion.optimizer} configuration achieved test RMSE {champion.test_rmse_mean:.4f} kWh "
      f"and R2 {champion.test_r2_mean:.4f}. Persistence achieved RMSE {analysis['persistence_test']['rmse']:.4f} kWh. "
      f"The deployed fixed seed-7 model has test RMSE {seed_rmse:.4f} kWh; this is distinct from the family mean. "
      "Small differences between top methods are not sufficient evidence of statistically significant superiority.")
    p(f"{analysis['diverged_runs']} runs diverged across the complete plan. Baseline outcomes: {failure_text}. "
      "A pairing without three non-diverged runs in any configuration has no eligible final test score. All attempted settings remain in the logs.", "SmallReport")
    page()

    title(7, "Convergence, computation, and stability")
    image("convergence_effective_passes.png", 242)
    p("Figure 2. Identical-baseline linear runs, seed 7. The loss range focuses on non-diverged runs; failed traces outside that range are explicitly marked. "
      "All raw values are in the CSV and HTML. The x-axis excludes additional monitoring. Loss uses standardized targets.", "SmallReport")
    image("convergence_elapsed_seconds.png", 242)
    p("Figure 3. The same objective against elapsed seconds, including monitoring. Timing is machine-dependent. "
      "L-BFGS may perform multiple closures per accepted step; its x-axis accounting therefore differs from the epoch plot.", "SmallReport")
    p("Stochastic updates can raise monitored loss; momentum can overshoot and adaptive coordinates do not guarantee monotonic descent. "
      "The logs retain the fraction of checkpoint increases, last-five loss SD, final gradient norm, and time to objective .35. "
      "A missing threshold time means the threshold was not observed, rather than a zero duration. Models return the validation-best checkpoint, not necessarily the final one.")
    page()

    title(8, "Sensitivity and forecast diagnostics")
    image("sensitivity.png", 306)
    p("Figure 4. One-factor validation sensitivity, mean +/- sample SD. Other fields are held at the relevant baseline. "
      "Seed is the repeated factor, not a second tuned hyperparameter. Scaling changes the optimization landscape and the meaning of a fixed ridge penalty.", "SmallReport")
    image("predictions.png", 205)
    p("Figure 5. Fixed-seed validation-selected champion, first held-out week. The figure uses actual test observations. "
      "The model is a one-hour-ahead predictor, not a perfectly smooth reconstruction of household peaks.", "SmallReport")
    p("Residual scatter and histograms are exported separately. They expose peak underprediction, asymmetry, or heteroskedasticity without changing the final selected model. "
      "The dashboard can generate 24-hour recursive forecasts by feeding predictions back into future feature rows; future observed values never enter those features. "
      "This extension has no separate multi-horizon accuracy claim.")
    page()

    title(9, "System, reproducibility, and conclusions")
    sub("Architecture and delivery")
    p("UCI download/cache -> validated hourly energy -> causal features -> chronological split -> train-only transformations -> "
      "shared model/objective -> instrumented optimizers -> validation selection -> final test -> saved model and exports. "
      "The seven-page Streamlit application exposes data analysis, interactive training, optimizer comparison, benchmark landscapes, historical forecasting, and reports.")
    sub("Benchmark interpretation")
    p("The playground includes positive-definite quadratic, condition-number-100 quadratic, Rosenbrock, and Himmelblau objectives. "
      "Contours and 3D surfaces share starting points; paths and normalized gradient arrows illustrate geometry. Exact-gradient deterministic functions "
      "make SGD and mini-batch GD identical to GD here. Multiple Himmelblau basins and Rosenbrock's curved valley demonstrate why non-convex guarantees differ from quadratic analysis.")
    sub("Reproduction and evidence")
    p("Run python -m pytest -q; python -m src.experiments --mode demo; python -m src.build_report --mode demo. "
      "The manifest records data hash, source checksum, partitions, seeds, software versions, and CPU/thread metadata. "
      "Repeated cached results preserve their original measured training times; report regeneration time is not a new training measurement. "
      "The Colab notebook performs the same commands after ZIP upload. Model bundles store fitted transforms and weights without training/test arrays.")
    sub("Limitations and next steps")
    p("The dataset describes one household, and the demo spans an early contiguous period. Three seeds quantify limited seed variability, not population confidence. "
      "One validation period permits temporal selection bias; a larger study should use rolling-origin validation, multiple seasons and households, "
      "equal tuning resources, and uncertainty estimates. RSS is a sampled whole-process footprint. CPU timing includes monitoring and depends on environment load. "
      + ("Full mode can be expensive, especially sample-wise ANN SGD; it is implemented but not represented as already measured here." if mode == "demo" else "The full run remains limited to the available dataset and protocol; broader temporal and household validation is still needed."))
    p(f"Within this measured budget, validation selected {champion['model']} / {champion.optimizer}; its test error improves on persistence. "
      "The broader conclusion is that curvature, stochasticity, preprocessing, and computational accounting all matter. "
      "The project supports defensible comparisons and documents failures rather than assuming a preferred optimizer.")
    sub("References")
    p("[1] Hebrail, G. and Berard, A. Individual Household Electric Power Consumption. UCI, 2006. DOI:10.24432/C58K54.<br/>"
      "[2] Kingma, D. P. and Ba, J. Adam: A Method for Stochastic Optimization. arXiv:1412.6980, 2014.<br/>"
      "[3] Nocedal, J. Updating Quasi-Newton Matrices with Limited Storage. Mathematics of Computation 35(151), 773-782, 1980. DOI:10.1090/S0025-5718-1980-0572855-7.<br/>"
      "[4] SciPy, minimize(method='L-BFGS-B'): docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html.<br/>"
      "[5] PyTorch, Optimization algorithms: docs.pytorch.org/docs/stable/optim.html.", "SmallReport")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d8e2eb"))
        canvas.line(58, 46, A4[0]-58, 46)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#53677b"))
        canvas.drawString(58, 31, "OptiEnergy AI | Measured " + mode + " study")
        canvas.drawRightString(A4[0]-58, 31, str(doc.page))
        canvas.restoreState()

    path = report_dir / "Project_Report.pdf"
    document = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=58, rightMargin=58,
                                 topMargin=48, bottomMargin=61, title="OptiEnergy AI - Optimization Study",
                                 author="OptiEnergy AI Project")
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    write_analysis(report_dir, directory, manifest, analysis, board, runs, chosen_cfg, seed_rmse)
    print(f"Built {path}")
    return path


def write_analysis(report_dir, directory, manifest, analysis, board, runs, chosen_cfg, seed_rmse):
    champion = board.iloc[0]
    rows = ["| Model | Optimizer | Val RMSE | Test RMSE ± SD | Test MAE | Test R² | Seconds ± SD |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in board.iterrows():
        rows.append(f"| {r['model']} | {r.optimizer} | {r.val_rmse_mean:.4f} | {r.test_rmse_mean:.4f} ± {r.test_rmse_std:.4f} | "
                    f"{r.test_mae_mean:.4f} | {r.test_r2_mean:.4f} | {r.training_seconds_mean:.4f} ± {r.training_seconds_std:.4f} |")
    failure_groups = runs[(runs.sweep == "baseline") & (runs.status == "diverged")].groupby(["model", "optimizer"]).size()
    failure_text = "; ".join(f"{model}/{optimizer}: {count} failed seeds" for (model, optimizer), count in failure_groups.items()) or "none"
    text = f"""# Measured experimental analysis

This is the **{manifest['mode']}** study: {manifest['max_hours'] or 'all'} hourly records before feature filtering, baseline {manifest['epochs']} epochs, seeds {manifest['seeds']}. {analysis['run_count']} runs were attempted; {analysis['diverged_runs']} diverged. Cached repetitions retain original measured timings. Dataset raw SHA-256: `{manifest['dataset']['raw_sha256']}`.

Chronological usable row counts: {manifest['split']['train']['rows']} train, {manifest['split']['validation']['rows']} validation, {manifest['split']['test']['rows']} test. Test origins: {manifest['split']['test']['first_origin']} through {manifest['split']['test']['last_origin']}.

The champion is **{champion['model']} / {champion.optimizer}**, chosen by mean validation RMSE **{champion.val_rmse_mean:.4f} kWh**. Final test RMSE is **{champion.test_rmse_mean:.4f} ± {champion.test_rmse_std:.4f} kWh**, sample SD. The fixed deployment seed 7 has RMSE {seed_rmse:.4f} kWh, which is not the family mean. Persistence test RMSE is {analysis['persistence_test']['rmse']:.4f} kWh.

The leading validation scores are close. Their order does not demonstrate statistical significance; another row can have better test RMSE without becoming the selected champion. Test observations never guide selection.

## Validation-selected leaderboard

{chr(10).join(rows)}

## What the measured study supports

* Baseline divergence: {failure_text}. Failed runs retain finite checkpoints for diagnostics but are ineligible for final selection. The full suite includes smaller learning-rate sweeps for all first-order pairings.
* The unregularized standardized Hessian has numerical rank {analysis['hessian']['numerical_rank']} of {analysis['hessian']['dimension']}. Its full condition number is undefined at tolerance; positive-spectrum condition number is {analysis['hessian']['positive_spectrum_condition_number']:.2f}.
* Ridge alpha=.01 raises the smallest eigenvalue to {analysis['hessian']['ridge_alpha_0.01_min_eigenvalue']:.5f}. Its finite full condition number {analysis['hessian']['ridge_alpha_0.01_condition_number']:.2f} is not directly comparable to the unregularized positive-spectrum-only number.
* The full-data GD step bound is 2/L={analysis['hessian']['safe_gd_lr_upper_bound']:.5f}. SGD divergence below that bound is possible because individual samples have higher curvature.
* The analytic gradient's central-difference relative error is {analysis['gradient_check_relative_error']:.3e}.
* Full-batch linear methods are deterministic across seeds; zero predictive SD is expected. ANN initialization and mini-batch order produce variability.

## One-factor sensitivity

"""
    for title, model, optimizer, parameter in [("Learning rate", "linear", "Adam", "lr"), ("Batch size", "linear", "Adam", "batch_size"),
            ("Momentum", "linear", "Momentum", "momentum"), ("Regularization", "ridge", "Adam", "alpha"),
            ("Scaling", "linear", "Adam", "scaling"), ("Schedule", "linear", "Adam", "schedule"), ("Epochs", "linear", "Adam", "epochs")]:
        group = one_factor(runs, model, optimizer, parameter)
        text += f"### {title}: {model} / {optimizer}\n\n| {parameter} | Validation RMSE mean | Sample SD | Valid seeds / attempts |\n|---|---:|---:|---:|\n"
        for _, r in group.iterrows():
            mean = f"{r['mean']:.4f}" if np.isfinite(r['mean']) else "not scored"
            sd = f"{r.sd:.4f}" if np.isfinite(r.sd) else "not available"
            text += f"| {r[parameter]} | {mean} | {sd} | {int(r.valid_seeds)} / {int(r.attempts)} |\n"
        text += "\n"
    text += """## Reading the graphs and counters

The fixed-baseline convergence graphs hold model, objective, initial weights, split, and baseline hyperparameters constant. The tuned leaderboard can vary one selected setting per pairing. These answer different questions; demo tuning coverage differs across pairings. Effective data passes, gradient calls, optimizer updates, and seconds must be interpreted together. Monitoring overhead is separately counted and included in seconds. Checkpoint stability and threshold time depend on monitoring frequency. Memory is checkpoint-sampled whole-process RSS.

A lower final training objective can generalize worse; returned models use validation-best checkpoints. `final_training_loss` and `final_gradient_norm` refer to the last finite checkpoint, not necessarily the restored model. `best_epoch` identifies the returned checkpoint. Missing threshold time means not reached. Failed runs remain in `runs.csv` and `baseline_summary.csv`; absence from the final leaderboard is deliberate.

## Limits

This study does not establish multi-household generalization or universal optimizer superiority. Three-seed SD is not a confidence interval. The supplied demo does not measure full-year performance; full mode must be executed separately for the complete history and sensitivity coverage. Recursive 24-hour forecasts are a causal extension with no separately claimed multi-horizon score. Reference data: Hebrail & Berard, UCI, DOI:10.24432/C58K54.
"""
    (report_dir / "Experimental_Analysis.md").write_text(text, encoding="utf-8")
    chosen = {k: chosen_cfg[k] for k in ("model", "optimizer", "lr", "batch_size", "epochs", "momentum", "alpha", "scaling", "schedule")}
    presentation = f"""# OptiEnergy AI - presentation content and speaker notes

**10 slides · 9 minutes total.** The measurements below are from the actual {manifest['mode']} run, not a proposed full experiment. Use the named exported graphs on the matching slides.

## 1. Title and research question (0:00-0:40)

**On slide:** OptiEnergy AI. Which optimization methods reach useful next-hour forecasts efficiently and stably?

**Speaker notes:** Electricity forecasting is the application, while optimization is the main research subject. I compare convex linear/ridge models and a non-convex ANN, holding conditions fixed within each family. I measure accuracy together with computation and stability.

## 2. Dataset and units (0:40-1:30)

**On slide:** UCI 235; {manifest['dataset']['raw_rows']:,} minute records; kWh = sum(kW)/60; {manifest['dataset']['complete_hours']:,} complete observed hours. Visual: `figures/dataset_eda.png`.

**Speaker notes:** Power and energy have different units. Only complete 60-minute hours form targets. At origin t, the previous hour is finished; I predict the next interval. Missing targets are never filled. Credit Hebrail and Berard, DOI:10.24432/C58K54, CC BY 4.0.

## 3. Leakage-safe features and splits (1:30-2:20)

**On slide:** 26 causal inputs; lags 1/3/6/12/24; rolling mean/std; cyclical calendar; chronological 70/15/15; boundary purge; train-only scaling.

**Speaker notes:** Forward fill is limited to six input hours. Every feature uses information available at the forecast origin. I purge labels crossing split boundaries, fit transformations on training rows, and test causality by changing future observations. The demo has {manifest['split']['train']['rows']} train, {manifest['split']['validation']['rows']} validation, and {manifest['split']['test']['rows']} test rows.

## 4. Objective and geometry (2:20-3:20)

**On slide:** J=||Aw-u||²/n + alpha*wᵀDw/2. Gradient=2Aᵀ(Aw-u)/n+alpha*Dw. Hessian=2AᵀA/n+alpha*D. GD: 0<eta<2/L.

**Speaker notes:** The Hessian is positive semidefinite, proving convexity. Redundant inputs make it singular; the numerical rank is {analysis['hessian']['numerical_rank']}/{analysis['hessian']['dimension']}. Ridge adds curvature without penalizing the bias. ANN parameters interact nonlinearly, so the global convex guarantee no longer applies. My gradient-check error is {analysis['gradient_check_relative_error']:.2e}.

## 5. Eight optimization methods (3:20-4:15)

**On slide:** Batch GD, SGD, mini-batch GD, Momentum, Nesterov, RMSprop, Adam, L-BFGS. Show two update equations plus state/cost table.

**Speaker notes:** The seven first-order variants are hand-written for linear models. Momentum stores direction, RMSprop stores squared gradients, and Adam stores first and second moments with bias correction. L-BFGS stores curvature pairs and uses a line search. I count every closure evaluation, because one accepted step may cost several gradients.

## 6. Fair experimental protocol (4:15-5:10)

**On slide:** Seeds 7/21/42; same initialization and split; equal example-processing budgets; separate updates/gradients/time; validation-only tuning; final test.

**Speaker notes:** An epoch is one data pass but has different update counts. Monitoring is separately counted, and timing includes it. One-factor sensitivity changes learning rate, batch size, momentum, regularization, schedule, epochs, or scaling. Demo sweeps representative pairings; full mode expands coverage. The direct sklearn solution is a reference rather than an iterative matched-budget method.

## 7. Measured results (5:10-6:15)

**On slide:** {analysis['run_count']} runs; {analysis['diverged_runs']} diverged. Validation selection: **{champion['model']} / {champion.optimizer}**. Test RMSE **{champion.test_rmse_mean:.3f} ± {champion.test_rmse_std:.3f} kWh**; persistence **{analysis['persistence_test']['rmse']:.3f} kWh**. Visual: `figures/leaderboard.png`.

**Speaker notes:** These are actual measurements on {manifest['max_hours'] or 'all'} clock hours, not all four years for demo mode. The lowest validation mean selects the champion; a better observed test score does not change selection. Close top scores do not establish significance. The deployable model uses fixed seed 7, whose score is {seed_rmse:.3f} kWh. Selection: {chosen}.

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
"""
    (report_dir / "Presentation_Content.md").write_text(presentation, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["demo", "full"], default="demo")
    build_report(parser.parse_args().mode)
