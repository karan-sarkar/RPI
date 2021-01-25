import matplotlib.pyplot as plt
import numpy as np
plt.scatter([1, -1], [0, 0], c = [1, -1])
x1, x2 = np.mgrid[-5:5:0.01, -5:5:0.01]
y_2 = (x1.ravel()).reshape((x1.shape[0], x2.shape[0]))
y_1 = (x1.ravel()**3 - x2.ravel()).reshape((x1.shape[0], x2.shape[0]))
plt.contour(x1, x2, y_1, [0])
plt.contour(x1, x2, y_2, [0])
plt.show()