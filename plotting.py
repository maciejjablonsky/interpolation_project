import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from splains import cubic_splains
from lagrange import lagrange

plots_path = 'plots/'
filename = 'Obiadek'
method_name = 'lagrange'


method = {
    'lagrange':lagrange,
    'splains':cubic_splains
}

files = ['chelm', 'MountEverest', 'SpacerniakGdansk', 'WielkiKanionKolorado', 'Redlujjj']

for filename in files:
    nodes_number = {
        'lagrange':[3,5, 10, 50],
        'splains': [3, 5, 10, 50]
        }
    nrows, ncols = 2,2
    fig = plt.figure(figsize=(15, 8), dpi=300)
    fig.suptitle('Interpolacja metodą Lagrange\'a. Zbiór danych %s' % filename)
    data = pd.read_csv('data/' + filename +'.csv')
    print('Computing %s %s' % (filename, method_name))
    plt.subplots_adjust(hspace=0.5)
    for i, nodes in enumerate(nodes_number[method_name]):
        idx = np.round(np.linspace(0, len(data) - 1, nodes).astype(int))
        X = np.array(data['distance'])[idx]
        Y = np.array(data['height'])[idx]
        x = np.linspace(min(X), max(X), 1000)
        x, y = method[method_name](x, X, Y)
        ax = fig.add_subplot(nrows, ncols, i +1)
        ax.grid()
        ax.plot(X, Y, 'o', label='Węzły podane do interpolacji')
        ax.plot(np.array(data['distance']), np.array(data['height']), label='Wartości rzeczywiste')
        ax.plot(x,y, label='Krzywa po interpolacji')
        ax.set_ylabel('Wysokość [m]')
        ax.set_xlabel('Dystans [m]')
        ax.title.set_text('Interpolacja przy %d węzłach' % len(X))
        ax.legend()
    print('Plotting %s %s' % (filename, method_name))
    fig.tight_layout(rect=[0, 0, 1, 0.95])    
    fig.savefig(plots_path + method_name + '_' + filename + '.png', bbox='tight') 
