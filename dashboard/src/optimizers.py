"""First-order NumPy optimizer classes. Nesterov evaluates the gradient at lookahead()."""
import numpy as np


class GradientDescent:
    def __init__(self, lr=.01, momentum=.9):
        if not np.isfinite(lr) or lr <= 0 or not 0 <= momentum < 1:
            raise ValueError("Invalid optimizer hyperparameters")
        self.lr, self.beta, self.t = lr, momentum, 0

    def lookahead(self, weights):
        return weights

    def step(self, weights, gradient):
        self.t += 1
        return weights - self.lr * gradient


class Momentum(GradientDescent):
    def step(self, weights, gradient):
        self.t += 1
        if not hasattr(self, "velocity"):
            self.velocity = np.zeros_like(weights)
        self.velocity = self.beta * self.velocity - self.lr * gradient
        return weights + self.velocity


class Nesterov(Momentum):
    def lookahead(self, weights):
        return weights + self.beta * getattr(self, "velocity", np.zeros_like(weights))


class RMSprop(GradientDescent):
    def step(self, weights, gradient):
        self.t += 1
        self.square = .99 * getattr(self, "square", np.zeros_like(weights)) + .01 * gradient ** 2
        return weights - self.lr * gradient / (np.sqrt(self.square) + 1e-8)


class Adam(GradientDescent):
    def step(self, weights, gradient):
        self.t += 1
        self.first = .9 * getattr(self, "first", np.zeros_like(weights)) + .1 * gradient
        self.second = .999 * getattr(self, "second", np.zeros_like(weights)) + .001 * gradient ** 2
        first = self.first / (1 - .9 ** self.t)
        second = self.second / (1 - .999 ** self.t)
        return weights - self.lr * first / (np.sqrt(second) + 1e-8)


def make_optimizer(name, lr=.01, momentum=.9):
    cls = {"Batch GD": GradientDescent, "SGD": GradientDescent, "Mini-batch GD": GradientDescent,
           "Momentum": Momentum, "Nesterov": Nesterov, "RMSprop": RMSprop, "Adam": Adam}.get(name)
    if cls is None:
        raise ValueError("L-BFGS uses a function/gradient closure; it is handled by SciPy/PyTorch")
    return cls(lr, momentum)
