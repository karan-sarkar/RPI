import keras
from keras import layers
from keras import backend as K
import numpy as np

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

from sklearn.svm import SVC

vals = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
best_model = None
best_score = 0
best_val = 0
for val in vals:
    model = SVC(C = val, kernel = 'poly', coef0 = 1, degree = 8)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_model = model
        best_val = val

print(1-best_score)
print(best_val)

import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c = y)
x1, x2 = np.mgrid[-5:5:0.01, -5:5:0.01]
y_hat = best_model.decision_function(np.c_[x1.ravel(), x2.ravel()]).reshape((x1.shape[0], x2.shape[0]))
plt.contour(x1, x2, y_hat, [0])
plt.show()



