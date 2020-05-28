import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce


def phi(x_samples: np.ndarray, i: int) -> np.float64:
    def product(x): return reduce(lambda a, b: a * b, x)
    nominator = product(list(np.poly1d([1, -x_sample])
                         for k, x_sample in enumerate(x_samples) if i != k))
    denominator = product([x_samples[i] - x_sample for k,
                           x_sample in enumerate(x_samples) if i != k])
    return nominator / denominator


def lagrange(x_to_interpolate, x_samples: np.ndarray, y_samples: np.ndarray) -> np.ndarray:
    phis = list(map(lambda i: phi(x_samples, i), range(len(x_samples))))
    return np.array(list(map(lambda x: np.sum([y * phis[i](x) for i, y in enumerate(y_samples)]), x_to_interpolate)))


data = pd.read_csv('data/MountEverest.csv')
X = np.array(data['distance'])
Y = np.array(data['height'])
1
x = np.linspace(0, max(X), 500)
step = 70
plt.plot(X[0::step], Y[0::step], 'x', color='red')
l = lagrange(x, X[0::step], Y[0::step])
plt.plot(x, l)
plt.show()
