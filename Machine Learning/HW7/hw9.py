TRAIN_FILE = 'zip.train'
TEST_FILE = 'zip.test'

import numpy as np
X1 = np.loadtxt(TRAIN_FILE)
X2 = np.loadtxt(TEST_FILE)
X = np.vstack((X1, X2))

y = X[:, 0]
X = np.delete(X, 0, axis=1)

y[y != 1] = -1

from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='full')
pca.fit(X)
X = pca.transform(X)

X = (X - np.min(X, axis = 0)) / (np.max(X, axis = 0) - np.min(X, axis = 0))

from sklearn.model_selection import train_test_split
X_test, X_train, y_test, y_train = train_test_split(X, y, test_size= 300 / X.shape[0])

from numpy.polynomial.legendre import legvander2d
Z_train = legvander2d(X_train[:, 0], X_train[:, 1], [8, 8])
Z_train = Z_train[:, [i for i in range(81) if int(i / 9) + (i % 9) <= 8]]
Z_test = legvander2d(X_test[:, 0], X_test[:, 1], [8, 8])
Z_test = Z_test[:, [i for i in range(81) if int(i / 9) + (i % 9) <= 8]]
print(Z_train.shape)

from sklearn.linear_model import Ridge
clf = Ridge(alpha=0.01)
clf.fit(Z_train, y_train)
print(1 - clf.score(Z_test, y_test))


import matplotlib.pyplot as plt
y_hat_test = clf.predict(Z_test)
print(np.mean((y_hat_test - y_test)**2))
'''

plt.scatter(X[:, 0], X[:, 1], c = y)
x1, x2 = np.mgrid[-5:5:0.01, -5:5:0.01]
print(x1.shape)
z = legvander2d(x1.ravel(), x2.ravel(), [8, 8])
z = z[:, [i for i in range(81) if int(i / 9) + (i % 9) <= 8]]
print(z.shape)
y_hat = clf.predict(z).reshape((x1.shape[0], x2.shape[0]))
print(y_hat.shape)
plt.contour(x1, x2, y_hat, [0])

'''

import tqdm

alpha = np.linspace(0, 2, num = 200)
cv = np.zeros(200)
err = np.zeros(200)
for i in tqdm.tqdm(range(alpha.shape[0])):
    for j in range(Z_train.shape[0]):
        clf = Ridge(alpha = i)
        clf.fit(np.delete(Z_train, j, axis=0), np.delete(y_train, j))
        pred = clf.predict(Z_train[j, :].reshape(1, 45))
        cv[i] += (y_train[j] - pred)**2 / Z_train.shape[0]
    
    clf = Ridge(alpha=i)
    clf.fit(Z_train, y_train)
    y_hat_test = clf.predict(Z_test)
    err[i] = np.mean((y_hat_test - y_test)**2)

plt.plot(alpha, cv)
plt.plot(alpha, err)
plt.legend(['E_cv', 'E_test'])


plt.show()