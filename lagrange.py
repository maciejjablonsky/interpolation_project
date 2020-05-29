import numpy as np
from functools import reduce


def phi(x_samples: np.ndarray, i: int) -> np.float64:
    """Computes base function for lagrange interpolation"""
    def product(x): return reduce(lambda a, b: a * b, x)
    nominator = product([np.poly1d([1, -x_sample])
                         for k, x_sample in enumerate(x_samples) if i != k])
    denominator = product([x_samples[i] - x_sample for k,
                           x_sample in enumerate(x_samples) if i != k])
    return nominator / denominator


def lagrange(x_to_interpolate: np.ndarray, x_samples: np.ndarray, y_samples: np.ndarray) -> tuple:
    """Interpolates f(x_to_interpolate) using lagrange polynomial interpolation given f as set of [x_samples, y_samples] points"""
    phis = [phi(x_samples, i) for i in range(len(x_samples))]
    return x_to_interpolate, np.array([sum([y * phis[i](x) for i, y in enumerate(y_samples)]) for x in x_to_interpolate])
