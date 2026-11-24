"""Central, serializable configuration. All paths are relative to this repository."""
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = ROOT / "saved_models"
OPTIMIZERS = ("Batch GD", "SGD", "Mini-batch GD", "Momentum", "Nesterov", "RMSprop", "Adam", "L-BFGS")
SEEDS = (7, 21, 42)


@dataclass(frozen=True)
class TrainConfig:
    model: str = "linear"
    optimizer: str = "Adam"
    lr: float = 0.01
    batch_size: int = 64
    epochs: int = 12
    momentum: float = 0.9
    alpha: float = 0.0
    schedule: str = "constant"
    decay: float = 0.05
    seed: int = 7
    threshold: float = 0.35
    scaling: str = "standard"

    def validate(self):
        if self.model not in {"linear", "ridge", "ann", "sklearn"}:
            raise ValueError("Unknown model")
        if self.optimizer not in OPTIMIZERS and not (self.model == "sklearn" and self.optimizer == "Direct solve"):
            raise ValueError("Unknown optimizer")
        if not 0 < self.lr <= 10 or not 0 <= self.momentum < 1:
            raise ValueError("Require 0 < learning rate <= 10 and 0 <= momentum < 1")
        if self.batch_size < 1 or self.epochs < 1 or self.alpha < 0 or self.decay < 0:
            raise ValueError("Invalid batch size, epochs, regularization, or decay")
        if self.model in {"linear", "sklearn"} and self.alpha != 0:
            raise ValueError("Use model='ridge' for a regularized linear model")
        if self.schedule not in {"constant", "decay"}:
            raise ValueError("Unknown learning-rate schedule")
        if self.scaling not in {"standard", "minmax", "robust", "none"}:
            raise ValueError("Unknown feature scaling")
        return self

    def to_dict(self):
        return asdict(self)
