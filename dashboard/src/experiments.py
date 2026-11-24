"""Reproducible baseline + one-variable-at-a-time studies, validation selection, final test."""
import argparse
import hashlib
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import MODELS, OPTIMIZERS, RESULTS, SEEDS, TrainConfig
from src.data_loader import load_hourly
from src.evaluation import regression_metrics
from src.models import LinearModel, gradient_check, linear_hessian
from src.prediction import save_bundle
from src.preprocessing import prepare_data
from src.training import train
from src.utils import environment, fingerprint, write_json

ENGINE_VERSION = "1.0.0"


def experiment_plan(mode="demo", epochs=10, sensitivity=True):
    plan = []
    for model in ("linear", "ridge", "ann"):
        for optimizer in OPTIMIZERS:
            base = TrainConfig(model=model, optimizer=optimizer, epochs=epochs,
                               alpha=.01 if model == "ridge" else 0.)
            plan.append(("baseline", base))
            if not sensitivity:
                continue
            # Demo exercises every sensitivity dimension on representative models.
            # Full applies each relevant dimension to every model/optimizer pairing.
            if optimizer != "L-BFGS" and (mode == "full" or (model == "linear" and optimizer == "Adam")):
                for lr in (.0001, .001, .01, .1):
                    plan.append(("learning_rate", replace(base, lr=lr)))
            if optimizer not in {"Batch GD", "SGD", "L-BFGS"} and (mode == "full" or (model == "linear" and optimizer == "Adam")):
                for size in (16, 32, 64, 128):
                    plan.append(("batch_size", replace(base, batch_size=size)))
            if optimizer in {"Momentum", "Nesterov"} and (mode == "full" or (model == "linear" and optimizer == "Momentum")):
                for momentum in (.5, .9, .99):
                    plan.append(("momentum", replace(base, momentum=momentum)))
            if model in {"ridge", "ann"} and (mode == "full" or (model == "ridge" and optimizer == "Adam")):
                for alpha in (0., .001, .01, .1):
                    plan.append(("regularization", replace(base, alpha=alpha)))
            if optimizer != "L-BFGS" and (mode == "full" or (model == "linear" and optimizer == "Adam")):
                plan.append(("schedule", replace(base, schedule="decay")))
            if mode == "full" or (model == "linear" and optimizer == "Adam"):
                for scaling in ("standard", "minmax", "robust", "none"):
                    plan.append(("scaling", replace(base, scaling=scaling)))
                for count in (5, 10, 20) if mode == "demo" else (10, 30, 60):
                    plan.append(("epochs", replace(base, epochs=count)))
    plan.append(("reference", TrainConfig(model="sklearn", optimizer="Direct solve", epochs=1)))
    # Store the same configuration once; sweep tables recover baseline rows explicitly.
    unique = {}
    for sweep, cfg in plan:
        unique.setdefault(fingerprint(cfg.to_dict()), (sweep, cfg))
    return list(unique.values())


def run_experiments(mode="demo", output=None, epochs=None, max_hours=None, sensitivity=True, use_cache=True):
    if mode not in {"demo", "full"}:
        raise ValueError("mode must be demo or full")
    epochs = epochs or (10 if mode == "demo" else 60)
    max_hours = max_hours if max_hours is not None else (2400 if mode == "demo" else None)
    output = Path(output or RESULTS / mode)
    output.mkdir(parents=True, exist_ok=True)
    cache = RESULTS / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    runtime = environment()
    hourly, metadata = load_hourly()
    selected_hourly = hourly.iloc[:max_hours] if max_hours else hourly
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(selected_hourly, index=True).values.tobytes()).hexdigest()
    prepared = {}

    def get_data(scaling):
        if scaling not in prepared:
            prepared[scaling] = prepare_data(hourly, scaling, max_hours)
        return prepared[scaling]

    base_data = get_data("standard")
    write_json(output / "manifest.json", {"engine_version": ENGINE_VERSION, "mode": mode,
        "epochs": epochs, "max_hours": max_hours, "seeds": SEEDS, "data_hash": data_hash,
        "dataset": metadata, "environment": runtime, "sensitivity": sensitivity,
        "split": {k: {"rows": len(f), "first_origin": str(f.index.min()), "last_origin": str(f.index.max())}
                  for k, f in base_data.frames.items()},
        "protocol": "70/15/15 chronological; boundary labels purged; train-only scaling; validation-only tuning; final test once per selected model/optimizer and seed",
        "budget": "first order: epochs*n example-gradient budget; L-BFGS: at most epochs full-data closure calls; monitoring separately counted; sklearn direct solve not budget-matched"})
    plan = experiment_plan(mode, epochs, sensitivity)
    runs, histories, objects = [], [], {}
    started = time.perf_counter()
    for number, (sweep, base) in enumerate(plan, 1):
        for seed in SEEDS:
            cfg = replace(base, seed=seed)
            data = get_data(cfg.scaling)
            configuration = cfg.to_dict()
            group = configuration.copy()
            group.pop("seed")
            config_id = fingerprint(group)
            run_id = fingerprint({"config": configuration, "data": data_hash, "engine": ENGINE_VERSION,
                                  "environment": runtime})
            path = cache / (run_id + ".joblib")
            cache_hit = use_cache and path.exists()
            if cache_hit:
                result = joblib.load(path)
            else:
                result = train(data, cfg)
                joblib.dump(result, path)
            objects[run_id] = (result, data)
            row = {"run_id": run_id, "config_id": config_id, "sweep": sweep, "cache_hit": cache_hit,
                   **configuration, **result.summary}
            runs.append(row)
            history = result.history.assign(run_id=run_id, config_id=config_id, sweep=sweep,
                                             model=cfg.model, optimizer=cfg.optimizer, seed=seed)
            histories.append(history)
            print(f"[{number}/{len(plan)}] {cfg.model:7} {cfg.optimizer:13} seed={seed} {sweep:15} "
                  f"val_RMSE={result.summary['val_rmse']:.4f} {result.summary['status']}", flush=True)
    runs = pd.DataFrame(runs)
    runs.to_csv(output / "runs.csv", index=False)
    baseline = runs[runs.sweep.isin(["baseline", "reference"])]
    baseline.groupby(["model", "optimizer", "status"]).agg(
        repetitions=("seed", "count"), validation_rmse_mean=("val_rmse", "mean"),
        validation_rmse_std=("val_rmse", "std"), training_seconds_mean=("training_seconds", "mean")
    ).reset_index().to_csv(output / "baseline_summary.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(output / "histories.csv", index=False)
    # Only candidates with all three finite, non-diverged repetitions are eligible.
    eligible = runs[runs.status != "diverged"]
    candidates = eligible.groupby(["model", "optimizer", "config_id"]).agg(
        val_rmse_mean=("val_rmse", "mean"), seeds=("seed", "nunique")).reset_index()
    candidates = candidates[candidates.seeds == len(SEEDS)]
    chosen = candidates.sort_values(["val_rmse_mean", "config_id"]).groupby(["model", "optimizer"], sort=False).head(1)
    chosen.to_csv(output / "selected_configs.csv", index=False)
    final = []
    for _, candidate in chosen.iterrows():
        for _, run in runs[runs.config_id == candidate.config_id].iterrows():
            result, data = objects[run.run_id]
            prediction = data.inverse_target(result.model.predict(data.X_test))
            actual = data.inverse_target(data.y_test)
            metrics = regression_metrics(actual, prediction)
            final.append({**run.to_dict(), **{"test_" + k: v for k, v in metrics.items()}})
            save_bundle(output / "models" / f"{run.model}_{run.optimizer.replace(' ', '_')}_{run.seed}.joblib",
                        result, data, {"run_id": run.run_id, "data_hash": data_hash,
                                       "last_training_target": str(data.frames["train"].target_time.max())})
    final = pd.DataFrame(final)
    final.to_csv(output / "final_test_runs.csv", index=False)
    columns = ["val_rmse", "test_mse", "test_rmse", "test_mae", "test_r2", "training_seconds", "iterations",
               "gradient_evaluations", "function_evaluations", "effective_passes", "final_training_loss",
               "final_validation_loss", "final_gradient_norm", "loss_increase_fraction", "tail_loss_std",
               "peak_observed_process_rss_mib"]
    board = final.groupby(["model", "optimizer"])[columns].agg(["mean", "std"])
    board.columns = ["_".join(c) for c in board.columns]
    board = board.reset_index().sort_values("val_rmse_mean")
    board.insert(0, "validation_rank", range(1, len(board) + 1))
    board.to_csv(output / "leaderboard.csv", index=False)
    # Champion family/config uses validation means; seed is fixed in advance to 7.
    champion = chosen.iloc[0]
    champion_run = runs[(runs.config_id == champion.config_id) & (runs.seed == SEEDS[0])].iloc[0]
    result, data = objects[champion_run.run_id]
    save_bundle(MODELS / f"champion_{mode}.joblib", result, data,
                {"selection": "lowest mean validation RMSE, fixed deployment seed 7", "data_hash": data_hash,
                 "training_last_target": str(data.frames["train"].target_time.max()), "max_hours": max_hours})
    shutil.copy2(MODELS / f"champion_{mode}.joblib", output / "champion.joblib")
    pred = data.frames["test"][["target_time"]].reset_index()
    pred["actual_kwh"] = data.inverse_target(data.y_test)
    pred["predicted_kwh"] = data.inverse_target(result.model.predict(data.X_test))
    pred.to_csv(output / "predictions.csv", index=False)
    # Persistence predicts that next hour equals the most recently observed hour.
    naive = regression_metrics(pred.actual_kwh, data.frames["test"].energy_current.to_numpy())
    hessian = linear_hessian(LinearModel.design(base_data.X_train))
    eigenvalues = np.linalg.eigvalsh(hessian)
    h_ridge = linear_hessian(LinearModel.design(base_data.X_train), .01)
    ridge_eigenvalues = np.linalg.eigvalsh(h_ridge)
    positive = eigenvalues[eigenvalues > eigenvalues.max() * 1e-10]
    numeric_rank = len(positive)
    write_json(output / "analysis.json", {
        "champion": champion.to_dict(), "champion_seed": SEEDS[0], "persistence_test": naive,
        "gradient_check_relative_error": gradient_check(np.zeros(hessian.shape[0]),
            LinearModel.design(base_data.X_train[:50]), base_data.y_train[:50]),
        "hessian": {"min_eigenvalue": float(eigenvalues.min()), "max_eigenvalue": float(eigenvalues.max()),
                    "numerical_rank": numeric_rank, "dimension": len(eigenvalues),
                    "condition_number": float(eigenvalues.max()/eigenvalues.min()) if numeric_rank == len(eigenvalues) else None,
                    "positive_spectrum_condition_number": float(positive.max()/positive.min()),
                    "safe_gd_lr_upper_bound": float(2/eigenvalues.max()),
                    "ridge_alpha_0.01_min_eigenvalue": float(ridge_eigenvalues.min()),
                    "ridge_alpha_0.01_condition_number": float(ridge_eigenvalues.max()/ridge_eigenvalues.min())},
        "run_count": len(runs), "diverged_runs": int((runs.status == "diverged").sum()),
        "cached_runs": int(runs.cache_hit.sum()), "newly_trained_runs": int((~runs.cache_hit).sum()),
        "total_wall_seconds": time.perf_counter() - started,
        "caveats": ["Demo covers an early contiguous subset, not all four years."] if mode == "demo" else []})
    from src.visualization import export_figures
    export_figures(output)
    print(f"Saved measured outputs to {output}", flush=True)
    return board


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["demo", "full"], default="demo")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-hours", type=int)
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    run_experiments(args.mode, epochs=args.epochs, max_hours=args.max_hours,
                    sensitivity=not args.skip_sensitivity, use_cache=not args.no_cache)
