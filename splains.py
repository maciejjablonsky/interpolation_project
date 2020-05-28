import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/MountEverest.csv')

# X = np.array([1, 3, 5])
# Y = np.array([6, -2, 4])
X = np.array(data['distance'][0::10])
Y = np.array(data['height'][0::10])

def splain(x, X, Y):
    n = len(X) -1
    A = np.empty((0, 4 * n), dtype=np.float64)
    b = []
    for i, (x_l, y_l, x_r, y_r)  in enumerate(zip(X, Y, X[1:], Y[1:])):
        h = x_r - x_l

        left = np.zeros((4*n), dtype=np.float64)
        left[4*i:4*i + 4] = [1, 0, 0, 0]
        A = np.vstack((A, left))
        b.append(y_l)
        
        right = np.zeros((4*n), dtype=np.float64)
        right[4*i:4*i + 4] = [h**i for i in range(4)]
        A = np.vstack((A, right))
        b.append(y_r)

        if i < n - 1:
            # first derivative
            der = np.zeros((4*n), dtype=np.float64)
            der[4*i:4*i + 8] = [0, 1, 2*h, 3*h**2, 0, -1, 0, 0]
            A = np.vstack((A, der))
            b.append(0)
            # second derivate
            dder = np.zeros((4*n), dtype=np.float64)
            dder[4*i:4*i + 8] = [0, 0, 2, 6*h, 0, 0, - 2, 0]
            A= np.vstack((A, dder))
            b.append(0)
            
    left_edge = np.zeros((4*n), dtype=np.float64)
    left_edge[0:4] = [0, 0, X[1] - X[0], 0]
    right_edge = np.zeros((4*n), dtype=np.float64)
    right_edge[4*n - 4: 4*n] = [0, 0, 2, 6*(X[n] - X[n - 1])]
    A = np.vstack((A, left_edge))
    b.append(0)
    A = np.vstack((A, right_edge))
    b.append(0)

    b = np.array(b).T
    coefficients = np.linalg.solve(A, b).reshape((-1, 4))
    polynomials = [np.poly1d(c[::-1]) for c in coefficients]
    print(polynomials)

    y = []
    i = 0
    for x_ in x:
        if x_ > X[i + 1]:
            i += 1
        y.append(polynomials[i](x_ - X[i]))
    return y

# f = lambda x:np.exp(-x/6) * np.sin(x)
f = lambda x: abs(x)
# x = np.linspace(min(X), max(X), 2*len(data))
X = np.linspace(-5, 5, num=101)
Y = f(X)
x = np.linspace(-5, 5, num=1000)
y = splain(x, X, Y)
plt.plot(x,y, color='red', label='Interpolated')
plt.plot(X, f(X), '.', color='green', label='Real data')
plt.legend()
plt.show()