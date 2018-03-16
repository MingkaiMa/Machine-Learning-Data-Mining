import matplotlib.pyplot as plt
import numpy as np

N = 100
data_x = np.arange(N)
rdm = (np.random.rand(N) - 0.5)
data_y1 = data_x + rdm * 10
data_y2 = np.sqrt(data_x) + rdm
data_y3 = data_x ** 2 + rdm * 1000
plt.scatter(data_x, data_y3, color = 'green')
plt.show()
