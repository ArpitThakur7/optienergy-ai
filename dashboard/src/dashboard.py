"""Shared Streamlit views. Training only occurs after an explicit form submission."""
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import OPTIMIZERS, RESULTS, ROOT, TrainConfig
from src.benchmark_functions import BENCHMARKS
from src.data_loader import load_hourly
from src.evaluation import regression_metrics
from src.feature_engineering import features_at_origins, supervised_frame
from src.prediction import forecast, load_bundle
from src.preprocessing import prepare_data
from src.training import train
from src.visualization import TEMPLATE, landscape, line, trajectory


@st.cache_data(show_spinner="Loading cached hourly UCI data…")
def dataset():
    return load_hourly()


def result_dir():
    return RESULTS / st.session_state.get("results_mode", "demo")


def measured(name):
    path = result_dir() / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def heading(kicker, title, subtitle):
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def missing_results():
    st.info("No measured results for this mode yet. Run: python -m src.experiments --mode " +
            st.session_state.get("results_mode", "demo"))


def overview():
    heading("RESEARCH / OVERVIEW", "Better forecasts. Measured optimization.",
            "A controlled study of convex linear models and a non-convex neural network.")
    hourly, meta = dataset()
    board = measured("leaderboard.csv")
    a, b, c, d = st.columns(4)
    a.metric("Minute observations", f"{meta['raw_rows']:,}")
    b.metric("Complete hourly targets", f"{meta['complete_hours']:,}")
    c.metric("Optimizers", "8")
    if not board.empty:
        champion = board.iloc[0]
        d.metric("Validation-selected test RMSE", f"{champion.test_rmse_mean:.3f} kWh")
        st.success(f"Lowest mean validation RMSE: {champion['model']} / {champion.optimizer}. "
                   "Ranking is determined by validation data; the displayed test score is final evaluation.")
    else:
        d.metric("Repeated seeds", "3")
        missing_results()
    st.plotly_chart(line(hourly.energy_kwh.resample("D").sum(min_count=24).rename("Daily energy (kWh)").reset_index(),
                         "origin", "Daily energy (kWh)", title="Daily consumption · only fully observed days"), use_container_width=True)
    manifest = result_dir() / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        st.info(f"Displayed run: {m['mode']} · {m['max_hours'] or 'all'} hourly records · baseline {m['epochs']} epochs · seeds {m['seeds']}. "
                "Demo results describe this subset and budget only.")
    left, right = st.columns(2)
    left.markdown("### What this project investigates\nLearning speed, noisy updates, conditioning, regularization, generalization, and the cost of each optimizer.")
    right.markdown("### Forecast contract\nAt origin **t**, use readings available through **t** to estimate energy in **[t, t+1 hour)**. Labels are never imputed.")


def dataset_analysis():
    heading("DATA / QUALITY", "Understand the signal", "Power is integrated into energy: hourly kWh = sum of 60 minute kW readings / 60.")
    hourly, meta = dataset()
    a, b = st.columns(2)
    a.metric("Missing / invalid minutes", f"{meta['missing_or_invalid_minutes']:,}")
    b.metric("Incomplete hours excluded as targets", f"{meta['incomplete_hours']:,}")
    st.plotly_chart(line(hourly.reset_index(), "origin", "energy_kwh", title="Observed hourly energy (kWh)"), use_container_width=True)
    missing = (60 - hourly.observed_minutes).resample("ME").sum().rename("Missing minutes").reset_index()
    st.plotly_chart(px.bar(missing, x="origin", y="Missing minutes", template=TEMPLATE, title="Missingness over time"), use_container_width=True)
    features = supervised_frame(hourly)
    left, right = st.columns(2)
    with left:
        pattern = hourly.energy_kwh.groupby((hourly.index.hour - 1) % 24).mean().rename("Mean kWh").rename_axis("hour").reset_index()
        st.plotly_chart(px.bar(pattern, x="hour", y="Mean kWh", template=TEMPLATE, title="Hourly pattern (descriptive only)"), use_container_width=True)
    with right:
        pattern = hourly.energy_kwh.groupby((hourly.index - pd.Timedelta(hours=1)).dayofweek).mean().rename("Mean kWh").rename_axis("weekday").reset_index()
        st.plotly_chart(px.bar(pattern, x="weekday", y="Mean kWh", template=TEMPLATE, title="Weekday pattern · Monday = 0"), use_container_width=True)
    columns = ["energy_current", "lag_1", "lag_3", "lag_6", "lag_12", "lag_24", "rolling_mean_24", "target"]
    st.plotly_chart(px.imshow(features[columns].corr(), zmin=-1, zmax=1, color_continuous_scale="RdBu_r", template=TEMPLATE,
                              title="Correlation · descriptive full history, never used for tuning"), use_container_width=True)
    feature = st.selectbox("Feature distribution", columns)
    st.plotly_chart(px.histogram(features, x=feature, nbins=70, template=TEMPLATE), use_container_width=True)
    st.caption("Timestamps are naive local-clock values supplied by UCI. No unverified DST localization is applied. Partial hours, negative power, invalid numbers, and duplicate timestamps are handled explicitly.")


def model_training():
    heading("MODELS / TRAIN", "Run a controlled training job", "Metrics below use validation data. Use the experiment CLI for the sealed final test protocol.")
    with st.form("training_controls"):
        a, b, c = st.columns(3)
        model = a.selectbox("Model", ["linear", "ridge", "ann", "sklearn"])
        optimizer = b.selectbox("Optimizer", OPTIMIZERS, index=6)
        scaling = c.selectbox("Scaling", ["standard", "minmax", "robust", "none"])
        lr = a.select_slider("Learning rate", options=[.0001, .001, .01, .1], value=.01)
        size = b.selectbox("Batch size", [16, 32, 64, 128], index=2)
        epochs = c.number_input("Epochs", min_value=1, max_value=200, value=10)
        momentum = a.selectbox("Momentum", [.5, .9, .99], index=1)
        alpha = b.selectbox("L2 strength (ridge / ANN)", [0., .001, .01, .1])
        schedule = c.selectbox("Schedule", ["constant", "decay"])
        seed = a.selectbox("Seed", [7, 21, 42])
        hours = b.selectbox("Data budget (hours)", [2400, 6000, "All"])
        submit = st.form_submit_button("Train model", type="primary")
    if model in {"linear", "sklearn"} and alpha:
        st.info("L2 is available for ridge and ANN; linear/reference runs use alpha=0.")
    if optimizer == "L-BFGS":
        st.caption("L-BFGS uses full-data closures and a line search. The learning rate, batch size, momentum, and schedule controls do not apply. Its closure budget is the epoch count.")
    if model == "sklearn":
        st.caption("The sklearn reference uses a direct least-squares solver; optimizer controls do not apply.")
    if submit:
        try:
            hourly, _ = dataset()
            data = prepare_data(hourly, scaling, None if hours == "All" else hours)
            cfg = TrainConfig(model=model, optimizer=optimizer, lr=lr, batch_size=size, epochs=int(epochs),
                              momentum=momentum, alpha=alpha if model in {"ridge", "ann"} else 0.,
                              seed=seed, schedule=schedule, scaling=scaling)
            key = (tuple(sorted(cfg.to_dict().items())), hours)
            if st.session_state.get("training_key") != key:
                bar, plot = st.progress(0), st.empty()

                def update(epoch, total, history):
                    bar.progress(min(1., epoch / total))
                    plot.plotly_chart(line(history, "epoch", ["train_loss", "validation_loss"], title="Live standardized loss"), use_container_width=True)

                result = train(data, cfg, update)
                bar.progress(1.)
                st.session_state["training_result"] = result
                st.session_state["training_data"] = data
                st.session_state["training_key"] = key
            else:
                st.info("Reused this session's identical training run.")
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
    if "training_result" in st.session_state:
        result, data = st.session_state.training_result, st.session_state.training_data
        a, b, c, d = st.columns(4)
        a.metric("Validation RMSE", f"{result.summary['val_rmse']:.4f} kWh")
        b.metric("Validation MAE", f"{result.summary['val_mae']:.4f} kWh")
        c.metric("Validation R²", f"{result.summary['val_r2']:.3f}")
        d.metric("Training seconds", f"{result.summary['training_seconds']:.2f}")
        st.plotly_chart(line(result.history, "epoch", ["train_loss", "validation_loss"], title="Loss at monitored checkpoints"), use_container_width=True)
        st.plotly_chart(line(result.history, "epoch", "gradient_norm", title="Full-training gradient norm"), use_container_width=True)
        actual = data.inverse_target(data.y_val)
        predicted = data.inverse_target(result.model.predict(data.X_val))
        frame = pd.DataFrame({"Actual (kWh)": actual, "Predicted (kWh)": predicted,
                              "Residual (kWh)": actual - predicted}, index=data.frames["validation"].index).reset_index()
        st.plotly_chart(line(frame, "origin", ["Actual (kWh)", "Predicted (kWh)"], title="Validation predictions"), use_container_width=True)
        st.plotly_chart(px.scatter(frame, x="Predicted (kWh)", y="Residual (kWh)", template=TEMPLATE, title="Validation residuals"), use_container_width=True)
        if hasattr(result.model, "weights"):
            # Standardized effect sizes comparable across feature scaling choices.
            scale = getattr(data.scaler, "scale_", np.ones(len(data.feature_names)))
            if data.scaling == "minmax":
                raw_coef = result.model.weights[:-1] * scale
            else:
                raw_coef = result.model.weights[:-1] / scale
            effects = raw_coef * data.frames["train"][data.feature_names].std(ddof=0).to_numpy() * data.y_std
            importance = pd.DataFrame({"feature": data.feature_names, "effect_kwh_per_sd": effects}).sort_values("effect_kwh_per_sd")
            st.plotly_chart(px.bar(importance, x="effect_kwh_per_sd", y="feature", orientation="h", template=TEMPLATE,
                                   title="Coefficients per training feature SD · associations, not causality"), use_container_width=True)
        st.json(result.summary, expanded=False)
        st.download_button("Export training history", result.history.to_csv(index=False), "training_history.csv", "text/csv")


def optimizer_comparison():
    heading("EXPERIMENTS / COMPARE", "Compare evidence, not assumptions", "Equal data-processing budgets; optimizer updates and closure costs are recorded separately.")
    board, runs, history = measured("leaderboard.csv"), measured("runs.csv"), measured("histories.csv")
    if board.empty:
        missing_results()
        return
    model = st.selectbox("Model family", sorted(runs.model.unique()))
    st.dataframe(board[board.model == model], use_container_width=True, hide_index=True)
    with st.expander("All baseline attempts, including failures"):
        st.dataframe(measured("baseline_summary.csv"), use_container_width=True, hide_index=True)
    st.caption("Table configurations are selected by mean validation RMSE across seeds. Baseline curves below hold all settings fixed within the model family.")
    view = history[(history.model == model) & (history.sweep.isin(["baseline", "reference"]))]
    seed = st.selectbox("Trajectory seed", [7, 21, 42])
    view = view[view.seed == seed]
    x = st.selectbox("Convergence budget axis", ["effective_passes", "epoch", "elapsed_seconds", "gradient_evaluations", "updates"])
    st.plotly_chart(line(view, x, "train_loss", "optimizer", "Training objective · common axes"), use_container_width=True)
    st.plotly_chart(line(view, x, "validation_loss", "optimizer", "Validation MSE · common axes"), use_container_width=True)
    st.plotly_chart(line(view, x, "gradient_norm", "optimizer", "Gradient norm"), use_container_width=True)
    subset = board[board.model == model]
    metric = st.selectbox("Final metric", ["test_rmse", "test_mae", "test_r2", "training_seconds"])
    st.plotly_chart(px.bar(subset, x="optimizer", y=metric+"_mean", error_y=metric+"_std", template=TEMPLATE,
                          title="Final selected configurations · mean ± sample SD"), use_container_width=True)
    st.subheader("One-variable sensitivity")
    optimizer = st.selectbox("Sensitivity optimizer", sorted(runs[runs.model == model].optimizer.unique()))
    parameter = st.selectbox("Vary one parameter", ["lr", "batch_size", "momentum", "alpha", "epochs", "scaling", "schedule"])
    frame = runs[(runs.model == model) & (runs.optimizer == optimizer)].copy()
    if frame.empty:
        return
    baseline = frame[frame.sweep.isin(["baseline", "reference"])].iloc[0]
    fields = ["lr", "batch_size", "momentum", "alpha", "epochs", "scaling", "schedule"]
    for field in fields:
        if field != parameter:
            frame = frame[frame[field] == baseline[field]]
    frame["eligible_val_rmse"] = frame.val_rmse.where(frame.status != "diverged")
    grouped = frame.groupby(parameter).agg(mean_val_rmse=("eligible_val_rmse", "mean"), sd_val_rmse=("eligible_val_rmse", "std"),
                                           valid_seeds=("eligible_val_rmse", "count"), attempts=("seed", "size")).reset_index()
    if len(grouped) < 2:
        st.info("This demo has no sensitivity sweep for that pairing. Full mode covers all applicable pairings.")
    else:
        st.plotly_chart(px.bar(grouped, x=parameter, y="mean_val_rmse", error_y="sd_val_rmse", template=TEMPLATE,
                              title="Validation RMSE (kWh) · other variables held at baseline"), use_container_width=True)
        st.dataframe(grouped, use_container_width=True, hide_index=True)
        st.caption("Diverged runs have no sensitivity score; valid-seed counts disclose missing repetitions.")
        heat = frame.pivot_table(index="seed", columns=parameter, values="eligible_val_rmse", aggfunc="mean")
        st.plotly_chart(px.imshow(heat, template=TEMPLATE, text_auto=".3f", aspect="auto", title="Validation RMSE by seed and parameter"), use_container_width=True)
    st.caption("Monitoring adds full-data evaluations beyond the training budget. RSS is the observed whole-process footprint, not optimizer-only allocation. Threshold crossing is checked at monitoring checkpoints; a blank time means not reached.")


def optimization_playground():
    heading("THEORY / PLAYGROUND", "See the optimization landscape", "Deterministic benchmark gradients isolate geometry from mini-batch noise.")
    a, b = st.columns([1, 3])
    with a:
        name = st.selectbox("Function", list(BENCHMARKS))
        selected = st.multiselect("Algorithms", OPTIMIZERS, default=["Batch GD", "Momentum", "Adam", "L-BFGS"])
        lr = st.number_input("Step size", min_value=.00001, max_value=1., value=BENCHMARKS[name][3], format="%.5f", key="benchmark_lr_"+name)
        momentum = st.slider("Momentum coefficient", 0., .99, .9)
        steps = st.slider("Maximum steps", 10, 1000, 150)
        start = BENCHMARKS[name][2]
        x0 = st.number_input("Starting x", value=float(start[0]), key="x0_"+name)
        y0 = st.number_input("Starting y", value=float(start[1]), key="y0_"+name)
        three_d = st.toggle("3D surface")
        animate = st.toggle("Animate first trajectory")
    paths = {opt: trajectory(name, opt, lr, steps, (x0, y0), momentum) for opt in selected}
    figure = landscape(name, paths, three_d, animate)
    with b:
        st.plotly_chart(figure, use_container_width=True)
    st.download_button("Export interactive landscape", figure.to_html(include_plotlyjs="cdn"), "landscape.html", "text/html")
    st.caption("Arrows show normalized negative gradients, not momentum directions. All algorithms see exact gradients here, so SGD and mini-batch GD reduce to ordinary GD. L-BFGS chooses its step using a line search; the step-size control does not apply. Trajectories stop before non-finite or extreme values, and may leave the displayed bounds.")
    explanations = {"Quadratic": "Positive-definite Hessian: GD converges for 0 < learning rate < 2/L. Different curvatures give different contraction rates.",
        "Ill-conditioned quadratic": "The Hessian eigenvalues are 1 and 100. The stable GD range is (0, 0.02); slow progress in the flat direction illustrates conditioning.",
        "Rosenbrock": "A curved narrow valley couples coordinates. Small loss does not necessarily mean the iterate has reached the minimizer (1, 1).",
        "Himmelblau": "Several global minimizers share zero loss. Starting points and optimizer dynamics determine the basin reached; no global guarantee follows."}
    st.info(explanations[name])


def energy_prediction():
    heading("FORECAST / ENERGY", "Forecast from a historical origin", "One-step forecasts use observed history. Recursive horizons feed predictions back as history.")
    path = result_dir() / "champion.joblib"
    options = (["Measured champion"] if path.exists() else []) + (["Session-trained model"] if "training_result" in st.session_state else [])
    if not options:
        missing_results()
        return
    choice = st.selectbox("Forecast model", options)
    if choice == "Measured champion":
        bundle = load_bundle(path)
    else:
        result, data = st.session_state.training_result, st.session_state.training_data
        bundle = {"model": result.model, "config": result.config,
                  "transform": {"feature_names": data.feature_names, "scaler": data.scaler, "y_mean": data.y_mean, "y_std": data.y_std},
                  "metadata": {"training_last_target": str(data.frames["train"].target_time.max()),
                               "max_hours": st.session_state.training_key[1]}}
    hourly, _ = dataset()
    features = features_at_origins(hourly).dropna()
    last_train = bundle.get("metadata", {}).get("training_last_target")
    # Also stay beyond validation used for checkpoint/config selection.
    hours = bundle.get("metadata", {}).get("max_hours", 2400)
    partition = prepare_data(hourly, bundle["config"]["scaling"], None if hours in {None, "All"} else int(hours))
    lower = max(pd.Timestamp(last_train), partition.frames["test"].index.min()) if last_train else partition.frames["test"].index.min()
    origins = features.index[features.index >= lower]
    if len(origins) == 0:
        st.warning("No valid held-out forecast origins")
        return
    date = st.date_input("Historical date", value=origins.min().date(), min_value=origins.min().date(), max_value=origins.max().date())
    available = origins[origins.date == date]
    if not len(available):
        st.info("No complete feature history on this date. Select another date.")
        return
    origin = st.selectbox("Origin (end of latest observed hour)", list(available), format_func=lambda t: str(t))
    horizon = st.select_slider("Horizon (hours)", options=[1, 6, 12, 24])
    try:
        output = forecast(bundle, hourly, origin, horizon)
    except ValueError as exc:
        st.error(str(exc))
        return
    a, b = st.columns(2)
    a.metric("Next-hour energy", f"{output.predicted_kwh.iloc[0]:.3f} kWh")
    actual = output.actual_kwh.iloc[0]
    b.metric("Observed energy", "Unavailable" if pd.isna(actual) else f"{actual:.3f} kWh")
    st.dataframe(output, hide_index=True, use_container_width=True)
    st.plotly_chart(line(output, "target_time", ["actual_kwh", "predicted_kwh"], title="Energy in intervals ending at the displayed timestamps"), use_container_width=True)
    st.download_button("Export forecast CSV", output.to_csv(index=False), "energy_forecast.csv", "text/csv")
    if horizon > 1:
        st.info("Only the first prediction uses a fully observed feature history. Later horizons use previous predictions; actual future readings are shown for evaluation only. Recursive performance has not been separately tuned.")
    st.caption("Negative model outputs are displayed as produced, consistently with evaluation. They indicate model error rather than physically valid energy.")


def reports_and_results():
    heading("EXPORT / RESULTS", "Take the evidence with you", "CSV data, interactive graphs, the academic report, and reproducible run metadata.")
    directory = result_dir()
    if not directory.exists():
        missing_results()
        return
    files = [p for p in directory.rglob("*") if p.is_file() and p.suffix in {".csv", ".json", ".png", ".html"}]
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, path.relative_to(directory))
    st.download_button("Download measured results + graphs", archive.getvalue(), "OptiEnergy_results.zip", "application/zip", type="primary")
    selection = st.selectbox("Individual result", [str(p.relative_to(directory)) for p in files]) if files else None
    if selection:
        path = directory / selection
        st.download_button("Download selected file", path.read_bytes(), path.name)
        if path.suffix == ".csv":
            st.dataframe(pd.read_csv(path).head(100), use_container_width=True)
    for name in ["Project_Report.pdf", "Presentation_Content.md", "Viva_Questions.md", "Mathematical_Documentation.md", "Experimental_Analysis.md"]:
        path = ROOT / "reports" / name
        if path.exists():
            st.download_button(name.replace("_", " "), path.read_bytes(), path.name)
    manifest = directory / "manifest.json"
    if manifest.exists():
        st.json(json.loads(manifest.read_text()), expanded=False)
