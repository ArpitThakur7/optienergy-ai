"""Linear MSE and ridge objectives share one analytically differentiated implementation."""
import numpy as np
import torch
from torch import nn


class LinearModel:
    def __init__(self, n_features):
        self.weights = np.zeros(n_features + 1, dtype=np.float64)

    @staticmethod
    def design(X):
        return np.column_stack([X, np.ones(len(X))])

    def predict(self, X):
        return self.design(X) @ self.weights


class RidgeModel(LinearModel):
    """Model form is identical; the trainer adds alpha/2 times squared non-bias weights."""


def linear_objective(weights, X, y, alpha=0.0):
    error = X @ weights - y
    penalty = weights.copy()
    penalty[-1] = 0.0
    loss = np.mean(error ** 2) + alpha * .5 * np.dot(penalty, penalty)
    gradient = 2 * X.T @ error / len(y) + alpha * penalty
    return float(loss), gradient


def linear_hessian(X, alpha=0.0):
    regularizer = np.eye(X.shape[1])
    regularizer[-1, -1] = 0
    return 2 * X.T @ X / len(X) + alpha * regularizer


def gradient_check(weights, X, y, alpha=.1, epsilon=1e-6):
    analytic = linear_objective(weights, X, y, alpha)[1]
    numerical = np.empty_like(weights)
    for j in range(len(weights)):
        step = np.zeros_like(weights)
        step[j] = epsilon
        numerical[j] = (linear_objective(weights + step, X, y, alpha)[0] -
                        linear_objective(weights - step, X, y, alpha)[0]) / (2 * epsilon)
    return float(np.linalg.norm(analytic - numerical) / max(1., np.linalg.norm(analytic), np.linalg.norm(numerical)))


class SmallANN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 32), nn.Tanh(),
                                 nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, X):
        return self.net(X).squeeze(-1)

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self(torch.as_tensor(X, dtype=torch.float32)).numpy()


def ann_objective(model, X, y, alpha):
    penalty = sum((p ** 2).sum() for name, p in model.named_parameters() if "weight" in name)
    return torch.mean((model(X) - y) ** 2) + alpha * .5 * penalty
