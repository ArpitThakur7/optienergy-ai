import numpy as np


def quadratic(p):
    x, y = p
    return .5 * (x*x + 4*y*y), np.array([x, 4*y])


def ill_conditioned(p):
    x, y = p
    return .5 * (x*x + 100*y*y), np.array([x, 100*y])


def rosenbrock(p):
    x, y = p
    return (1-x)**2 + 100*(y-x*x)**2, np.array([-2*(1-x)-400*x*(y-x*x), 200*(y-x*x)])


def himmelblau(p):
    x, y = p
    a, b = x*x+y-11, x+y*y-7
    return a*a+b*b, np.array([4*x*a+2*b, 2*a+4*y*b])


BENCHMARKS = {"Quadratic": (quadratic, (-4., 4.), (3., 3.), .1),
              "Ill-conditioned quadratic": (ill_conditioned, (-4., 4.), (3., 3.), .01),
              "Rosenbrock": (rosenbrock, (-2., 2.), (-1.2, 1.), .001),
              "Himmelblau": (himmelblau, (-5., 5.), (-3., -3.), .01)}
