from dataclasses import replace

import numpy as np
import pytest

from config import OPTIMIZERS, TrainConfig
from src.prediction import forecast, load_bundle, save_bundle
from src.training import train


@pytest.mark.parametrize("model", ["linear", "ridge", "ann"])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_finite_training_and_budget(data, model, optimizer):
    cfg = TrainConfig(model=model, optimizer=optimizer, epochs=2, alpha=.01 if model == "ridge" else 0)
    result = train(data, cfg)
    assert np.isfinite(result.summary["val_rmse"])
    assert result.summary["processed_examples"] <= cfg.epochs*len(data.y_train)
    assert result.summary["gradient_evaluations"] == result.summary["function_evaluations"]
    assert result.summary["best_validation_loss"] <= result.history.validation_loss.iloc[0] + 1e-7


def test_reproducibility(data):
    cfg = TrainConfig(model="ann", optimizer="Adam", epochs=3)
    first, second = train(data, cfg), train(data, cfg)
    np.testing.assert_array_equal(first.model.predict(data.X_val), second.model.predict(data.X_val))
    np.testing.assert_array_equal(first.history.train_loss, second.history.train_loss)


def test_direct_solver_reference(data):
    result = train(data, TrainConfig(model="sklearn"))
    assert result.summary["status"] == "direct_solution"
    assert result.summary["final_gradient_norm"] < 1e-6


def test_model_round_trip_and_recursive_no_leak(data, hourly, tmp_path):
    result = train(data, TrainConfig(epochs=3))
    save_bundle(tmp_path / "model.joblib", result, data)
    bundle = load_bundle(tmp_path / "model.joblib")
    np.testing.assert_array_equal(result.model.predict(data.X_test), bundle["model"].predict(data.X_test))
    origin = hourly.index[400]
    first = forecast(bundle, hourly, origin, 24)
    modified = hourly.copy()
    modified.loc[modified.index > origin, "energy_kwh"] = 9999
    second = forecast(bundle, modified, origin, 24)
    np.testing.assert_array_equal(first.predicted_kwh, second.predicted_kwh)


def test_test_labels_are_never_read_by_trainer(data):
    cfg = TrainConfig(epochs=2)
    first = train(data, cfg)
    data.y_test[:] = 99999
    second = train(data, cfg)
    np.testing.assert_array_equal(first.model.weights, second.model.weights)


def test_validation_checkpoint_restored(data):
    result = train(data, TrainConfig(optimizer="SGD", epochs=3, lr=.1))
    val = np.mean((result.model.predict(data.X_val) - data.y_val)**2)
    assert val == pytest.approx(result.summary["best_validation_loss"])
