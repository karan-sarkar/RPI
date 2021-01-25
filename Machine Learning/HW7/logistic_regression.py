TRAIN_FILE = 'zip.train'
TEST_FILE = 'zip.test'
EPOCHS = 100
LEARNING_RATE = 0.01

import numpy as np
X_train = np.loadtxt(TRAIN_FILE)
X_test = np.loadtxt(TEST_FILE)

y_train = X_train[:, 0]
y_test = X_test[:, 0]
X_train = np.delete(X_train, 0, axis=1)
X_test = np.delete(X_test, 0, axis=1)

X_train = X_train[np.where((y_train == 1) | (y_train == 5))]
y_train = y_train[np.where((y_train == 1) | (y_train == 5))]
X_test = X_test[np.where((y_test == 1) | (y_test == 5))]
y_test = y_test[np.where((y_test == 1) | (y_test == 5))]

y_train[y_train == 5] = 0
y_test[y_test == 5] = 0

from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='full')
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))


X_train = np.c_[X_train, X_train[:, 1] * X_train[:, 1]]
X_train = np.c_[X_train, X_train[:, 1] * X_train[:, 2]]
X_train = np.c_[X_train, X_train[:, 2] * X_train[:, 2]]
X_train = np.c_[X_train, X_train[:, 1] * X_train[:, 1] * X_train[:, 1]]
X_train = np.c_[X_train, X_train[:, 1] * X_train[:, 1] * X_train[:, 2]]
X_train = np.c_[X_train, X_train[:, 1] * X_train[:, 2] * X_train[:, 2]]
X_train = np.c_[X_train, X_train[:, 2] * X_train[:, 2] * X_train[:, 2]]

X_test = np.c_[X_test, X_test[:, 1] * X_test[:, 1]]
X_test = np.c_[X_test, X_test[:, 1] * X_test[:, 2]]
X_test = np.c_[X_test, X_test[:, 2] * X_test[:, 2]]
X_test = np.c_[X_test, X_test[:, 1] * X_test[:, 1] * X_test[:, 1]]
X_test = np.c_[X_test, X_test[:, 1] * X_test[:, 1] * X_test[:, 2]]
X_test = np.c_[X_test, X_test[:, 1] * X_test[:, 2] * X_test[:, 2]]
X_test = np.c_[X_test, X_test[:, 2] * X_test[:, 2] * X_test[:, 2]]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def prob(weights, x):
    return sigmoid(np.dot(x, weights))

def cost(weights, x, y):
    return  np.sum(y * np.log(prob(weights, x)) + (1 - y) * np.log(1 - prob(weights, x))) / (- x.shape[0])

def gradient(weights, x, y):
    return np.dot(x.T, prob(weights, x) - y) / x.shape[0]

weights = np.zeros(X_train.shape[1])
for _ in range(EPOCHS):
    for i in range(X_train.shape[0]):
        weights = weights - LEARNING_RATE * gradient(weights, X_train[i, :], y_train[i])




import matplotlib.pyplot as plt
plt.scatter(X_train[:, 1], X_train[:, 2], c = y_train)


x = np.linspace(-5,5,100)
y =  - weights[1] / weights[2] * x  - weights[0] / weights[2]
plt.plot(x, y, '-r', label='y=2x+1')



'''
y, x = np.ogrid[-10:10:100j, -10:10:100j]
plt.contour(x.ravel(), y.ravel(),  weights[0] + weights[1] * x + weights[2] * y + weights[3] * x * x + \
            weights[4] * x * y + weights[5] * y * y + weights[6] * x * x * x + weights[7] * x * x * y + \
            weights[8] * x * y * y + weights[9] * y * y * y , [0])
'''

plt.show()

from sklearn.metrics import accuracy_score
y = prob(weights, X_train)
y[y >= 0.5] = 1
y[y < 0.5] = 0

print(1 - accuracy_score(y_train, y))

print(X_train.shape, X_test.shape)
y = prob(weights, X_test)
y[y >= 0.5] = 1
y[y < 0.5] = 0

print(1- accuracy_score(y_test, y))