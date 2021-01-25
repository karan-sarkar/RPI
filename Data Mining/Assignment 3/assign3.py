import sys
TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]

LEARNING_RATE = float(sys.argv[4])
EPS = float(sys.argv[3])

import numpy as np
X_train = np.genfromtxt(TRAIN_FILE, delimiter = ',')
X_test = np.genfromtxt(TEST_FILE, delimiter = ',')

y_train = X_train[:, X_train.shape[1] - 1]
y_test = X_test[:, X_train.shape[1] - 1]
X_train = np.delete(X_train, X_train.shape[1] - 1, axis=1)
X_test = np.delete(X_test, X_train.shape[1] - 1, axis=1)

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def prob(weights, x):
    return sigmoid(np.dot(x, weights))

def cost(weights, x, y):
    return  np.sum(y * np.log(prob(weights, x)) + (1 - y) * np.log(1 - prob(weights, x))) / (- x.shape[0])

def gradient(weights, x, y):
    return np.dot(x.T, prob(weights, x) - y) / x.shape[0]

weights = np.zeros(X_train.shape[1])
old_weights = np.ones(X_train.shape[1])
while np.sum((weights - old_weights)**2) > EPS:
    old_weights = weights
    for i in range(X_train.shape[0]):
        weights = weights - LEARNING_RATE * gradient(weights, X_train[i, :], y_train[i])

print(weights)


from sklearn.metrics import accuracy_score
y = prob(weights, X_train)
y[y >= 0.5] = 1
y[y < 0.5] = 0

print('TRAIN accuracy', accuracy_score(y_train, y))

y = prob(weights, X_test)
y[y >= 0.5] = 1
y[y < 0.5] = 0

print('TEST accuracy', accuracy_score(y_test, y))