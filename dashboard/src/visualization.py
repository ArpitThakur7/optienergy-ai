from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

from src.benchmark_functions import BENCHMARKS
from src.optimizers import make_optimizer

TEMPLATE = "plotly_dark"
COLORS = ["#43dfb0", "#6a9dff", "#ffc66b", "#ea8afd", "#ff6d87", "#70d3f5", "#d1d770", "#b7a8ff"]


def line(frame, x, y, color=None, title=None):
    fig = px.line(frame, x=x, y=y, color=color, title=title, template=TEMPLATE,
                  color_discrete_sequence=COLORS)
    fig.update_layout(margin=dict(l=20, r=20, t=45, b=25), legend_title_text="")
    return fig


def trajectory(name, optimizer, lr, steps=200, start=None, momentum=.9):
    function, _, default, _ = BENCHMARKS[name]
    point = np.array(start if start is not None else default, dtype=float)
    if not np.isfinite(point).all() or steps < 1:
        raise ValueError("Invalid starting point or step count")
    path = [point.copy()]
    if optimizer == "L-BFGS":
        result = minimize(lambda p: function(p), point, jac=True, method="L-BFGS-B",
                          callback=lambda p: path.append(p.copy()), options={"maxiter": steps})
        if not np.array_equal(path[-1], result.x):
            path.append(result.x)
    else:
        engine = make_optimizer(optimizer, lr, momentum)
        for _ in range(steps):
            with np.errstate(over="ignore", invalid="ignore"):
                _, gradient = function(engine.lookahead(point))
                next_point = engine.step(point, gradient)
                value, _ = function(next_point)
            if not np.isfinite(value) or np.linalg.norm(next_point) > 1e6:
                break
            point = next_point
            path.append(point.copy())
    return np.array(path)


def landscape(name, paths, three_d=False, animate=False):
    function, bounds, _, _ = BENCHMARKS[name]
    axis = np.linspace(*bounds, 100)
    xx, yy = np.meshgrid(axis, axis)
    z, _ = function([xx, yy])
    fig = go.Figure()
    if three_d:
        fig.add_trace(go.Surface(x=xx, y=yy, z=z, opacity=.7, colorscale="Viridis", showscale=False))
    else:
        fig.add_trace(go.Contour(x=axis, y=axis, z=np.log1p(z), colorscale="Viridis",
                                 colorbar_title="log(1 + f)", contours_coloring="heatmap"))
    for i, (optimizer, path) in enumerate(paths.items()):
        values = np.array([function(p)[0] for p in path])
        common = dict(mode="lines+markers", name=optimizer, line=dict(color=COLORS[i % len(COLORS)]),
                      marker=dict(size=3))
        if three_d:
            fig.add_trace(go.Scatter3d(x=path[:, 0], y=path[:, 1], z=values, **common))
        else:
            fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], **common))
            # Show raw negative-gradient directions, normalized only for readability.
            for point in path[::max(1, len(path)//8)][:8]:
                g = function(point)[1]
                delta = -.25 * g / max(np.linalg.norm(g), 1e-12)
                fig.add_annotation(x=point[0]+delta[0], y=point[1]+delta[1], ax=point[0], ay=point[1],
                                   xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                                   arrowhead=2, arrowcolor=COLORS[i % len(COLORS)])
        for index, label, symbol in ((0, "start", "circle"), (-1, "end", "diamond")):
            p = path[index]
            kw = dict(x=[p[0]], y=[p[1]], mode="markers", name=optimizer+" "+label,
                      marker=dict(size=8, symbol=symbol, color=COLORS[i % len(COLORS)]))
            if three_d:
                fig.add_trace(go.Scatter3d(z=[values[index]], **kw))
            else:
                fig.add_trace(go.Scatter(**kw))
    if animate and paths and not three_d:
        path = next(iter(paths.values()))
        trace_index = len(fig.data)
        fig.add_trace(go.Scatter(x=[path[0, 0]], y=[path[0, 1]], mode="markers", marker=dict(size=13, color="white"), name="playhead"))
        fig.frames = [go.Frame(data=[go.Scatter(x=[p[0]], y=[p[1]])], traces=[trace_index], name=str(i))
                      for i, p in enumerate(path)]
        fig.update_layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Play trajectory", method="animate", args=[None, {"frame": {"duration": 35}, "fromcurrent": True}])])])
    fig.update_layout(template=TEMPLATE, title=name, height=540, xaxis_title="x", yaxis_title="y",
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x, y)"))
    if not three_d:
        fig.update_xaxes(range=list(bounds))
        fig.update_yaxes(range=list(bounds))
    return fig


def export_figures(result_dir):
    """Generate static publication figures and interactive HTML from measured CSV files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result_dir = Path(result_dir)
    out = result_dir / "figures"
    out.mkdir(exist_ok=True)
    history = pd.read_csv(result_dir / "histories.csv")
    # Identical baseline config and first seed: all optimizers on common axes.
    view = history[(history.model == "linear") & (history.sweep == "baseline") & (history.seed == 7)]
    for x, label in (("epoch", "Epoch / accepted L-BFGS update"), ("elapsed_seconds", "Elapsed seconds (includes monitoring)"),
                     ("gradient_evaluations", "Training gradient evaluations"), ("effective_passes", "Training example-gradient passes")):
        fig, ax = plt.subplots(figsize=(8.2, 4.2), layout="constrained")
        for name, group in view.groupby("optimizer", sort=False):
            ax.plot(group[x], group.train_loss, label=name)
        ax.set(xlabel=label, ylabel="Standardized MSE objective", title="Linear model: baseline convergence, seed 7")
        ax.set_yscale("log")
        statuses = pd.read_csv(result_dir / "runs.csv")
        failures = statuses[(statuses.model == "linear") & (statuses.sweep == "baseline") &
                            (statuses.seed == 7) & (statuses.status == "diverged")].optimizer.tolist()
        stable = view[~view.optimizer.isin(failures)]
        if failures and not stable.empty:
            ax.set_ylim(stable.train_loss.min() * .85, stable.train_loss.max() * 1.15)
            ax.text(.02, .96, "Outside displayed loss range: " + ", ".join(failures) + " (diverged)",
                    transform=ax.transAxes, va="top", fontsize=8, color="#a33333")
        ax.legend(fontsize=7, ncol=4)
        ax.grid(alpha=.2)
        fig.savefig(out / f"convergence_{x}.png", dpi=170)
        plt.close(fig)
        line(view, x, "train_loss", "optimizer", "Measured convergence").write_html(out / f"convergence_{x}.html", include_plotlyjs="cdn")
    leaderboard = pd.read_csv(result_dir / "leaderboard.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
    labels = leaderboard.model + " / " + leaderboard.optimizer
    ax.barh(labels, leaderboard.test_rmse_mean, xerr=leaderboard.test_rmse_std.fillna(0), color="#357eaa")
    ax.invert_yaxis()
    ax.set(xlabel="Test RMSE (kWh), mean +/- SD across 3 seeds", title="Validation-selected configurations")
    ax.tick_params(axis="y", labelsize=6)
    fig.savefig(out / "leaderboard.png", dpi=170)
    plt.close(fig)
    pred = pd.read_csv(result_dir / "predictions.csv", parse_dates=["origin", "target_time"])
    fig, ax = plt.subplots(figsize=(8.2, 3.5), layout="constrained")
    ax.plot(pred.target_time.iloc[:168], pred.actual_kwh.iloc[:168], label="Actual")
    ax.plot(pred.target_time.iloc[:168], pred.predicted_kwh.iloc[:168], label="Forecast", alpha=.8)
    ax.set(ylabel="Hourly energy (kWh)", xlabel="Target interval endpoint", title="Validation-selected champion: test forecast sample")
    ax.legend()
    fig.savefig(out / "predictions.png", dpi=170)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.3), layout="constrained")
    residual = pred.actual_kwh - pred.predicted_kwh
    axs[0].scatter(pred.predicted_kwh, residual, s=8, alpha=.5)
    axs[0].axhline(0, color="black", lw=1)
    axs[0].set(xlabel="Predicted (kWh)", ylabel="Actual - predicted (kWh)", title="Residuals")
    axs[1].hist(residual, bins=30)
    axs[1].set(xlabel="Residual (kWh)", ylabel="Count", title="Residual distribution")
    fig.savefig(out / "residuals.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2, 3.6), layout="constrained")
    for name, group in view.groupby("optimizer", sort=False):
        if name not in failures:
            ax.plot(group.effective_passes, group.gradient_norm, label=name)
    ax.set(xlabel="Training example-gradient passes", ylabel="Full objective gradient norm",
           title="Linear baseline: gradient norms of non-diverged runs")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=4)
    fig.savefig(out / "gradient_norms.png", dpi=170)
    plt.close(fig)
    for name in BENCHMARKS:
        paths = {opt: trajectory(name, opt, BENCHMARKS[name][3]) for opt in ("Batch GD", "Momentum", "Adam", "L-BFGS")}
        landscape(name, paths).write_html(out / (name.lower().replace(" ", "_") + ".html"), include_plotlyjs="cdn")
