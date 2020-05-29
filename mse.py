import numpy as np
from lagrange import lagrange
from splains import cubic_splines
import matplotlib.pyplot as plt
import pandas as pd

def mse(expected, evaluated): return (np.square(expected - evaluated)).mean()

plots_path = 'plots/'
method_name = 'splines'


method = {
    'lagrange':lagrange,
    'splines':cubic_splines
}

files = ['chelm', 'MountEverest', 'SpacerniakGdansk', 'WielkiKanionKolorado', 'Redlujjj']

for filename in files:
    print('Computing %s %s error'%(filename, method_name))
    max_nodes = 200
    nrows, ncols = 2,2
    data = pd.read_csv('data/' + filename +'.csv')
    err = []
    for nodes in range(2, max_nodes, 1):
        print('\rnodes: %d'% nodes, end='')
        x_real = np.array(data['distance'])
        y_real = np.array(data['height'])
        idx = np.round(np.linspace(0, len(data) - 1, nodes).astype(int))
        X = x_real[idx]
        Y = y_real[idx]
        _, y_interpolated = cubic_splines(x_real, X, Y)
        err.append(mse(y_real, y_interpolated))
    print('Plotting %s %s error' % (filename, method_name))
    fig = plt.figure()
    fig.suptitle('Średni błąd kwadratowy metody funkcji sklejanych\'a')
    plt.semilogy(range(2, max_nodes), err)
    plt.grid()
    plt.ylabel('Wartość błędu [m]')
    plt.xlabel('Ilość węzłów [n]')
    fig.savefig(plots_path + 'error_' + method_name + '_' + filename + '.png', dpi=200)
plt.show()
