import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from splains import cubic_splains
from lagrange import lagrange

plots_path = 'plots/'
filename = 'Obiadek'
method = 'lagrange'
data = pd.read_csv('data/' + filename +'.csv')

steps = [120, 85, 50, 35]
nrows, ncols = 2,2
fig = plt.figure()
fig.suptitle('Interpolacja metodą Lagrange\'a')

for i, step in enumerate(steps):
    X = np.array(data['distance'])[::step]
    Y = np.array(data['height'])[::step]
    x = np.linspace(min(X), max(X), 1000)
    y = lagrange(x, X, Y)
    ax = fig.add_subplot(nrows, ncols, i +1) 
    ax.plot(X, Y, '.', label='Punkty pomiarowe')
    ax.plot(x,y, label='Krzywa po interpolacji')
    ax.set_ylabel('Wysokość [m]')
    ax.set_xlabel('Dystans [m]')
    ax.title.set_text('Interpolacja przy %d węzłach' % len(X))
    ax.legend()
plt.tight_layout()
plt.show()
fig.savefig(plots_path + method + '_' + filename + '.png', dpi=200) 