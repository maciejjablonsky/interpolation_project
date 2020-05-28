import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing as tp


def splains(x_interpolated, X_samples, Y_samples) -> tuple:
    def fill_equall_offset_on_edges_equations(A: np.ndarray, b: np.ndarray, X: np.ndarray, Y: np.ndarray):
        n = len(X) - 1
        for i, (x_left, y_left, x_right, y_right) in enumerate(zip(X, Y, X[1:], Y[1:])):
            left_edge_equation = np.zeros((4*n), dtype=np.float64)
            left_edge_equation[4*i:4*i + 4] = [1, 0, 0, 0]
            right_edge_equation = np.zeros((4*n), dtype=np.float64)
            right_edge_equation[4*i:4*i +
                                4] = [(x_right - x_left)**i for i in range(4)]
            A = np.vstack((A, left_edge_equation, right_edge_equation))
            b = np.vstack((b, y_left, y_right))
        return A, b

    def fill_equally_continuous_curve_equations(A: np.ndarray, b: np.ndarray, X: np.ndarray, Y: np.ndarray) -> tuple:
        n = len(X) - 1
        for i, (x_left, y_left, x_right, y_right) in enumerate(zip(X[:-1], Y[:-1], X[1:-1], Y[1:-1])):
            dx = x_right - x_left
            derivative_equality_equation = np.zeros((4*n), dtype=np.float64)
            derivative_equality_equation[4*i:4*i +
                                         8] = [0, 1, 2*dx, 3*(dx**2), 0, -1, 0, 0]
            second_derivative_equality_equation = np.zeros(
                (4*n), dtype=np.float64)
            second_derivative_equality_equation[4 *
                                                i:4*i + 8] = [0, 0, 2, 6*dx, 0, 0, - 2, 0]
            A = np.vstack((A, derivative_equality_equation,
                           second_derivative_equality_equation))
            b = np.vstack((b, 0, 0))
        return A, b

    def fill_inflection_on_boundaries_equations(A: np.ndarray, b: np.ndarray, X: np.ndarray, Y: np.ndarray) -> tuple:
        n = len(X) - 1
        left_boundary = np.zeros((4*n), dtype=np.float64)
        left_boundary[0:4] = [0, 0, X[1] - X[0], 0]
        right_boundary = np.zeros((4*n), dtype=np.float64)
        right_boundary[4*n - 4: 4*n] = [0, 0, 2, 6*(X[n] - X[n - 1])]
        A = np.vstack((A, left_boundary, right_boundary))
        b = np.vstack((b, 0, 0))
        return A, b

    def compute_polynomials(A: np.ndarray, b: np.ndarray) -> list:
        coefficients = np.linalg.solve(A, b).reshape((-1, 4))
        return [np.poly1d(c[::-1]) for c in coefficients]

    def compute_y(x_in_intervals: np.ndarray, X_samples: np.ndarray, polynomials: tp.List[np.poly]):
        y = []
        interpolate = lambda x, i: polynomials[i](x-X_samples[i])
        for i, interval in enumerate(x_in_intervals):
            y.extend(map(lambda x:interpolate(x, i), interval))
        return y

    def reshape_into_intervals(x, X_samples):
        reshaped = []
        k = 0
        for i in range(len(X_samples) - 1):
            interval = []
            for dx in x[k:]:
                if dx <= X_samples[i + 1]:
                    interval.append(dx)
            k += len(interval)
            reshaped.append(interval)
        return reshaped

    A, b = np.empty((0, 4*(len(X_samples) - 1)),
                    dtype=np.float64), np.empty((0, 1), dtype=np.float64)
    A, b = fill_equall_offset_on_edges_equations(A, b, X_samples, Y_samples)
    A, b = fill_equally_continuous_curve_equations(A, b, X_samples, Y_samples)
    A, b = fill_inflection_on_boundaries_equations(A, b, X_samples, Y_samples)
    polynomials = compute_polynomials(A, b)
    intervals = reshape_into_intervals(x_interpolated, X_samples)
    y = compute_y(intervals, X_samples, polynomials)
    return x_interpolated, y


data = pd.read_csv('data/MountEverest.csv')

step = 10
X = np.array(data['distance'][0::step])
Y = np.array(data['height'][0::step])
x = np.linspace(min(X), max(X), num=1000)

x, y = splains(x, X, Y)
plt.plot(x, y, color='red', label='Interpolated')
plt.plot(X, Y, '.', color='green', label='Real data')
plt.legend()
plt.show()
