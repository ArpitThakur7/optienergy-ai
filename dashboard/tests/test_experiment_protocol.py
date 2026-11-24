from config import OPTIMIZERS, TrainConfig
from src.experiments import experiment_plan
from src.training import train


def test_full_sweeps_change_one_variable_at_a_time():
    mapping = {"learning_rate": "lr", "batch_size": "batch_size", "momentum": "momentum",
               "regularization": "alpha", "schedule": "schedule", "scaling": "scaling", "epochs": "epochs"}
    plan = experiment_plan("full", epochs=60)
    for sweep, config in plan:
        config.validate()
        if sweep in {"baseline", "reference"}:
            continue
        base = TrainConfig(model=config.model, optimizer=config.optimizer, epochs=60,
                           alpha=.01 if config.model == "ridge" else 0)
        changes = {key for key, value in config.to_dict().items() if base.to_dict()[key] != value}
        assert changes == {mapping[sweep]}
    for model in ("linear", "ridge", "ann"):
        assert {c.optimizer for sweep, c in plan if c.model == model and sweep == "baseline"} == set(OPTIMIZERS)
        for optimizer in OPTIMIZERS[:-1]:
            assert {c.lr for _, c in plan if c.model == model and c.optimizer == optimizer} >= {.0001, .001, .01, .1}


def test_divergence_preserves_finite_checkpoint(data):
    data.X_train[:, 0] *= 1e20
    result = train(data, TrainConfig(optimizer="Batch GD", lr=.1, epochs=2))
    assert result.summary["status"] == "diverged"
    assert result.summary["best_epoch"] == 0
    assert result.summary["message"]
