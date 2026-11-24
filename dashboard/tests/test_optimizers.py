import numpy as np
import pytest

from src.benchmark_functions import BENCHMARKS
from src.models import LinearModel, gradient_check, linear_hessian, linear_objective
from src.optimizers import Adam, Momentum, Nesterov, RMSprop, make_optimizer


def test_analytic_gradient_and_hessian():
    rng = np.random.default_rng(7)
    X = LinearModel.design(rng.normal(size=(50, 4)))
    y, w = rng.normal(size=50), rng.normal(size=5)
    assert gradient_check(w, X, y, .2) < 1e-8
    h = linear_hessian(X, .2)
    assert np.allclose(h, h.T)
    assert np.linalg.eigvalsh(h).min() > 0
    delta = rng.normal(size=5) * 1e-5
    assert np.allclose(linear_objective(w+delta, X, y, .2)[1] - linear_objective(w, X, y, .2)[1], h @ delta)


def test_intercept_is_not_regularized():
    w = np.array([2., 3.])
    X = np.ones((5, 2))
    g0 = linear_objective(w, X, np.ones(5), 0)[1]
    g1 = linear_objective(w, X, np.ones(5), .5)[1]
    np.testing.assert_allclose(g1-g0, [1., 0.])


def test_momentum_and_nesterov_steps():
    w = np.array([1., -2.])
    g = np.array([2., -1.])
    opt = Momentum(.1, .9)
    np.testing.assert_allclose(opt.step(w, g), [.8, -1.9])
    np.testing.assert_allclose(opt.step(np.array([.8, -1.9]), g), [.42, -1.71])
    nag = Nesterov(.1, .9)
    next_w = nag.step(w, g)
    np.testing.assert_allclose(nag.lookahead(next_w), [.62, -1.81])


def test_adaptive_first_step_equations():
    w, g = np.array([2., -1.]), np.array([3., -2.])
    np.testing.assert_allclose(Adam(.01).step(w, g), w - .01*g/(np.abs(g)+1e-8))
    np.testing.assert_allclose(RMSprop(.01).step(w, g), w - .01*g/(np.sqrt(.01*g*g)+1e-8))


@pytest.mark.parametrize("name", ["Batch GD", "SGD", "Mini-batch GD", "Momentum", "Nesterov", "RMSprop", "Adam"])
def test_all_first_order_methods_converge_on_convex_quadratic(name):
    w = np.array([3., -2.])
    opt = make_optimizer(name, .03)
    for _ in range(1500):
        point = opt.lookahead(w)
        gradient = np.array([point[0], 4*point[1]])
        w = opt.step(w, gradient)
    assert np.linalg.norm(w) < .04


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_benchmark_gradients(name):
    function = BENCHMARKS[name][0]
    p = np.array([.3, -.2])
    numerical = []
    for j in range(2):
        delta = np.eye(2)[j]*1e-6
        numerical.append((function(p+delta)[0]-function(p-delta)[0])/2e-6)
    np.testing.assert_allclose(function(p)[1], numerical, atol=1e-6)
