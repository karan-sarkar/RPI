from math import pi, sin, cos

EPOCHS = 50
LEARNING_RATE = 0.01

def partial_x(x, y):
    return 2 * x + 2 * y * y + 4 * pi * cos( 2 * pi * x) * sin (2 * pi * y)

def partial_y(x, y):
    return x * x + 4 * y + 4 * pi * sin ( 2 * pi * x) * cos (2 * pi * y)

def f(x, y):
    return x * x + 2 * y * y + 2 * sin (2 * pi * x) * sin (2 * pi * y)

x = -1
y = -1

f_vals = []

for _ in range(EPOCHS):
    try:
        f_vals.append(f(x, y))
        x -= LEARNING_RATE * partial_x(x, y)
        y -= LEARNING_RATE * partial_y(x, y)
        print(x, y, f(x, y))
    except:
        break

import matplotlib.pyplot as plt
import numpy as np

plt.plot(np.arange(len(f_vals)), np.array(f_vals))
plt.show()
