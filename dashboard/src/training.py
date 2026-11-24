"""Instrumented CPU training. Validation checkpoints, no test access in this module."""
import copy
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import psutil
import torch
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

from src.evaluation import regression_metrics
from src.models import LinearModel, RidgeModel, SmallANN, ann_objective, linear_objective
from src.optimizers import make_optimizer
from src.utils import seed_everything


class BudgetExhausted(Exception):
    """Expected termination when a quasi-Newton evaluation budget has been spent."""


@dataclass
class TrainingResult:
    model: object
    history: pd.DataFrame
    summary: dict
    config: dict


def train(data, config, progress=None):
    config.validate()
    seed_everything(config.seed)
    rng = np.random.default_rng(config.seed)
    n, d = data.X_train.shape
    X, y, V, v = data.X_train, data.y_train, data.X_val, data.y_val
    ann = config.model == "ann"
    if ann:
        X, y, V, v = [torch.as_tensor(z, dtype=torch.float32) for z in (X, y, V, v)]
        model = SmallANN(d)
    elif config.model == "sklearn":
        model = LinearRegression()
    else:
        model = (RidgeModel if config.model == "ridge" else LinearModel)(d)
        X, V = model.design(X), model.design(V)
    start = time.perf_counter()
    counters = {"updates": 0, "gradient_evaluations": 0, "function_evaluations": 0,
                "processed_examples": 0, "monitor_gradient_evaluations": 0,
                "monitor_function_evaluations": 0}
    rows, best, best_val, best_epoch = [], None, float("inf"), 0
    status, message, threshold_seconds = "budget_completed", "", None
    process = psutil.Process()
    peak_rss = process.memory_info().rss / 2 ** 20

    def state():
        if ann:
            return copy.deepcopy(model.state_dict())
        return copy.deepcopy(model) if config.model == "sklearn" else model.weights.copy()

    def record(epoch):
        nonlocal best, best_val, best_epoch, threshold_seconds, peak_rss
        if ann:
            model.zero_grad()
            objective = ann_objective(model, X, y, config.alpha)
            objective.backward()
            loss = float(objective.detach())
            grad = float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters())))
            with torch.no_grad():
                pred, vp = model(X).numpy(), model(V).numpy()
        elif config.model == "sklearn":
            pred, vp = model.predict(X), model.predict(V)
            w = np.r_[model.coef_, model.intercept_]
            loss, g = linear_objective(w, LinearModel.design(X), y)
            grad = float(np.linalg.norm(g))
        else:
            loss, g = linear_objective(model.weights, X, y, config.alpha)
            grad = float(np.linalg.norm(g))
            pred, vp = X @ model.weights, V @ model.weights
        counters["monitor_gradient_evaluations"] += 1
        counters["monitor_function_evaluations"] += 2
        val_mse = float(np.mean((np.asarray(vp) - data.y_val) ** 2))
        train_mse = float(np.mean((np.asarray(pred) - data.y_train) ** 2))
        if not np.isfinite([loss, val_mse, grad]).all() or loss > 1e12:
            raise FloatingPointError("Non-finite or exploding training objective")
        if val_mse < best_val:
            best, best_val, best_epoch = state(), val_mse, epoch
        elapsed = time.perf_counter() - start
        if loss <= config.threshold and threshold_seconds is None:
            threshold_seconds = elapsed
        peak_rss = max(peak_rss, process.memory_info().rss / 2 ** 20)
        row = {"epoch": epoch, "train_loss": loss, "train_mse": train_mse, "validation_loss": val_mse,
               "gradient_norm": grad, "elapsed_seconds": elapsed,
               "effective_passes": counters["processed_examples"] / n, **counters}
        rows.append(row)
        if progress:
            progress(epoch, config.epochs, pd.DataFrame(rows))

    # L-BFGS may need several full-data closure calls per accepted update.
    # Stop BEFORE a closure would exceed the same epochs*n example-gradient budget.
    budget = config.epochs * n
    try:
        if config.model == "sklearn":
            model.fit(X, y)
            status = "direct_solution"
            record(1)
        else:
            record(0)
            if config.optimizer == "L-BFGS":
                if ann:
                    optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=1,
                                                  max_eval=20, history_size=10, line_search_fn="strong_wolfe")

                    def closure():
                        if counters["processed_examples"] + n > budget:
                            raise BudgetExhausted()
                        optimizer.zero_grad()
                        loss = ann_objective(model, X, y, config.alpha)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite L-BFGS closure")
                        loss.backward()
                        counters["gradient_evaluations"] += 1
                        counters["function_evaluations"] += 1
                        counters["processed_examples"] += n
                        return loss

                    for epoch in range(1, config.epochs + 1):
                        optimizer.step(closure)
                        counters["updates"] += 1
                        record(epoch)
                else:
                    def objective(w):
                        if counters["processed_examples"] + n > budget:
                            raise BudgetExhausted()
                        value, gradient = linear_objective(w, X, y, config.alpha)
                        if not np.isfinite(value) or not np.isfinite(gradient).all():
                            raise FloatingPointError("Non-finite L-BFGS evaluation")
                        counters["gradient_evaluations"] += 1
                        counters["function_evaluations"] += 1
                        counters["processed_examples"] += n
                        return value, gradient

                    def callback(w):
                        model.weights = w.copy()
                        counters["updates"] += 1
                        record(counters["updates"])

                    result = minimize(objective, model.weights, method="L-BFGS-B", jac=True, callback=callback,
                                      options={"maxiter": config.epochs, "maxfun": config.epochs,
                                               "gtol": 1e-7, "ftol": 1e-12, "maxcor": 10})
                    model.weights = result.x.copy()
                    status = "converged" if result.success else "budget_completed"
                    message = str(result.message)
            else:
                if ann:
                    params = model.parameters()
                    if config.optimizer in {"Batch GD", "SGD", "Mini-batch GD"}:
                        optimizer = torch.optim.SGD(params, lr=config.lr)
                    elif config.optimizer in {"Momentum", "Nesterov"}:
                        # PyTorch's equivalent NAG formulation is documented in MATHEMATICS.md.
                        optimizer = torch.optim.SGD(params, lr=config.lr, momentum=config.momentum,
                                                     nesterov=config.optimizer == "Nesterov" and config.momentum > 0)
                    elif config.optimizer == "RMSprop":
                        optimizer = torch.optim.RMSprop(params, lr=config.lr, alpha=.99, eps=1e-8)
                    else:
                        optimizer = torch.optim.Adam(params, lr=config.lr, betas=(.9, .999), eps=1e-8)
                else:
                    optimizer = make_optimizer(config.optimizer, config.lr, config.momentum)
                size = n if config.optimizer == "Batch GD" else 1 if config.optimizer == "SGD" else config.batch_size
                for epoch in range(1, config.epochs + 1):
                    lr = config.lr / (1 + config.decay * (epoch - 1)) if config.schedule == "decay" else config.lr
                    if ann:
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                    else:
                        optimizer.lr = lr
                    order = rng.permutation(n)
                    for pos in range(0, n, size):
                        batch = order[pos:pos + size]
                        if ann:
                            optimizer.zero_grad()
                            objective = ann_objective(model, X[batch], y[batch], config.alpha)
                            if not torch.isfinite(objective) or objective > 1e12:
                                raise FloatingPointError("Exploding mini-batch loss")
                            objective.backward()
                            optimizer.step()
                        else:
                            point = optimizer.lookahead(model.weights)
                            objective, gradient = linear_objective(point, X[batch], y[batch], config.alpha)
                            if not np.isfinite(objective) or objective > 1e12 or not np.isfinite(gradient).all():
                                raise FloatingPointError("Exploding mini-batch loss")
                            model.weights = optimizer.step(model.weights, gradient)
                        counters["updates"] += 1
                        counters["gradient_evaluations"] += 1
                        counters["function_evaluations"] += 1
                        counters["processed_examples"] += len(batch)
                    record(epoch)
    except BudgetExhausted:
        status, message = "budget_completed", "Stopped before exceeding the example-gradient budget"
    except (FloatingPointError, OverflowError, RuntimeError) as exc:
        status, message = "diverged", str(exc)
    elapsed = time.perf_counter() - start
    if best is None:
        raise RuntimeError("No finite model checkpoint: " + message)
    if ann:
        model.load_state_dict(best)
    elif config.model == "sklearn":
        model = best
    else:
        model.weights = best
    vp = model.predict(data.X_val)
    metrics = regression_metrics(data.inverse_target(data.y_val), data.inverse_target(vp))
    history = pd.DataFrame(rows)
    changes = np.diff(history.train_loss.to_numpy())
    summary = {"status": status, "message": message, "training_seconds": elapsed,
               "iterations": counters["updates"], **counters,
               "effective_passes": counters["processed_examples"] / n,
               "best_epoch": int(best_epoch), "best_validation_loss": best_val,
               "final_training_loss": float(history.train_loss.iloc[-1]),
               "final_validation_loss": float(history.validation_loss.iloc[-1]),
               "final_gradient_norm": float(history.gradient_norm.iloc[-1]),
               "loss_increase_fraction": float(np.mean(changes > 0)) if len(changes) else 0.,
               "tail_loss_std": float(history.train_loss.tail(5).std(ddof=0)),
               "threshold_seconds": threshold_seconds,
               "threshold": config.threshold, "peak_observed_process_rss_mib": peak_rss,
               **{"val_" + k: value for k, value in metrics.items()}}
    return TrainingResult(model, history, summary, config.to_dict())
