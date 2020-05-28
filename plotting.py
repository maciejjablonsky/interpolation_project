import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from splains import cubic_splains
from lagrange import lagrange



plots_path = 'plots/'

filename = 'chelm'

data = pd.read_csv('data/' + filename +'.csv')

steps = [150, 100, 50, 30]
fig, ax = plt.subplots(2,2)
fig.suptitle('Interpolacja metodą Lagrange\'a')

X = np.array(data['distance'])[::steps[0]]
Y = np.array(data['height'])[::steps[0]]
x = np.linspace(min(X), max(X), 1000)
y = lagrange(x, X, Y)
ax[0,0].plot(X, Y, '.', label='Punkty pomiarowe')
ax[0,0].plot(x, y, label='Krzywa po interpolacji')
ax[0,0].legend()
ax[0,0].set_ylabel('Wysokość [m]')
ax[0,0].set_xlabel('Dystans [m]')
ax[0,0].title.set_text('Interpolacja przy %d węzłach' % (len(X)))

X = np.array(data['distance'])[::steps[1]]
Y = np.array(data['height'])[::steps[1]]
x = np.linspace(min(X), max(X), 1000)
y = lagrange(x, X, Y)
ax[0,1].plot(X, Y, '.', label='Punkty pomiarowe')
ax[0,1].plot(x,y, label='Krzywa po interpolacji')
ax[0,1].legend()
ax[0,1].set_ylabel('Wysokość [m]')
ax[0,1].set_xlabel('Dystans [m]')
ax[0,1].title.set_text('Interpolacja przy %d węzłach' % len(X))

X = np.array(data['distance'])[::steps[2]]
Y = np.array(data['height'])[::steps[2]]
x = np.linspace(min(X), max(X), 1000)
y = lagrange(x, X, Y)
ax[1,0].plot(X, Y, '.', label='Punkty pomiarowe')
ax[1,0].plot(x,y, label='Krzywa po interpolacji')
ax[1,0].legend()
ax[1,0].set_ylabel('Wysokość [m]')
ax[1,0].set_xlabel('Dystans [m]')
ax[1,0].title.set_text('Interpolacja przy %d węzłach' % len(X))

X = np.array(data['distance'])[::steps[3]]
Y = np.array(data['height'])[::steps[3]]
x = np.linspace(min(X), max(X), 1000)
y = lagrange(x, X, Y)
ax[1,1].plot(X, Y, '.', label='Punkty pomiarowe')
ax[1,1].plot(x,y, label='Krzywa po interpolacji')
ax[1,1].legend()
ax[1,1].set_ylabel('Wysokość [m]')
ax[1,1].set_xlabel('Dystans [m]')
ax[1,1].title.set_text('Interpolacja przy %d węzłach' % len(X))


plt.tight_layout()
plt.show()
fig.savefig(plots_path + filename + '.png', dpi=200) 