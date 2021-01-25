import sys
TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]
C = float(sys.argv[3])
EPS = float(sys.argv[4])
KERNEL = sys.argv[5]
SPREAD = float(sys.argv[6])


import numpy as np
X_train = np.genfromtxt(TRAIN_FILE, delimiter = ',')
X_test = np.genfromtxt(TEST_FILE, delimiter = ',')

y_train = np.copy(X_train[:, X_train.shape[1] - 1])
y_test = np.copy(X_test[:, X_test.shape[1] - 1])
X_train[:, X_train.shape[1] - 1] = 1
X_test[:, X_test.shape[1] - 1] = 1

K = np.zeros((X_train.shape[0], X_train.shape[0]))
if KERNEL == 'linear':
    K = X_train @ X_train.T
elif KERNEL == 'quadratic':
    K = (1 + X_train @ X_train.T)**2
elif KERNEL == 'gaussian':
    K = 2 * X_train @ X_train.T
    row_norms = np.sum(X_train**2, axis = 1)
    K = K - row_norms - row_norms[:, np.newaxis]
    K = np.exp(K / (2 * SPREAD * SPREAD))
    

eta = 1 / np.diag(K)
weights = np.ones((X_train.shape[0]))
new_weights = np.zeros((X_train.shape[0]))
while np.sum((weights - new_weights)**2) >= EPS:
    weights = new_weights
    for i in range(weights.shape[0]):
        new_weights[i] = new_weights[i] + eta[i] * (1 - y_train[i] * (np.dot(K[i, :], new_weights * y_train)))
        new_weights[new_weights < 0] = 0
        new_weights[new_weights > C] = C

K_hat = np.zeros((X_test.shape[0], X_train.shape[0]))
if KERNEL == 'linear':
    K_hat = X_test @ X_train.T
elif KERNEL == 'quadratic':
    K_hat = (1 + X_test @ X_train.T)**2
elif KERNEL == 'gaussian':
    K_hat = 2 * X_test @ X_train.T
    row_norms = np.sum(X_train**2, axis = 1)
    col_norms = np.sum(X_test**2, axis = 1)
    K_hat = (K_hat - row_norms) - col_norms[:, np.newaxis]
    K_hat = np.exp(K_hat / (2 * SPREAD * SPREAD))

y_hat = np.sign(K @ (weights * y_train))
print(np.mean(0.5 * y_hat * y_train + 0.5))
y_hat = np.sign(K_hat @ (weights * y_train))
print(np.mean(0.5 * y_hat * y_test + 0.5))
