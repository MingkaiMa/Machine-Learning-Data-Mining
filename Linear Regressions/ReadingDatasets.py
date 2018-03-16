import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


diabetes = datasets.load_diabetes()

fnum = 0
data_x = diabetes.data[:, np.newaxis, fnum]
data_y = diabetes.target

plt.scatter(data_x, data_y)
plt.show()
