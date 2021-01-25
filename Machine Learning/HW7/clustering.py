TRAIN_FILE = 'zip.train'
TEST_FILE = 'zip.test'

import numpy as np
X1 = np.loadtxt(TRAIN_FILE)
X2 = np.loadtxt(TEST_FILE)
X = np.vstack((X1, X2))

y = X[:, 0]
X = np.delete(X, 0, axis=1)