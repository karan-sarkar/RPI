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

from sklearn.neighbors import KNeighborsClassifier
k = np.arange(40) + 1
acc = np.zeros((40))
from sklearn.model_selection import cross_val_score
'''
for i in range(40):
    classifier = KNeighborsClassifier(n_neighbors=k[i])
    acc[i] = 1- cross_val_score(classifier, X_train, y_train, cv=100, scoring='accuracy').mean()
'''

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression

#('scaler', StandardScaler()), 
from math import sqrt
'''
for i in range(40):
    classifier = Pipeline([('clusters', KMeans(n_clusters = k[i], random_state= 0)),
                     ('gaussian', FunctionTransformer(lambda x : np.exp(-1 * np.square(x / (2 / sqrt(k[i]) ))))),
                     ('classifier', LogisticRegression(random_state=0, solver='lbfgs'))])
    acc[i] =  1- cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy').mean()
'''


import matplotlib.pyplot as plt

#plt.plot(k, acc)

classifier = Pipeline([('clusters', KMeans(n_clusters = 10, random_state= 0)),
                     ('gaussian', FunctionTransformer(lambda x : np.exp(-1 * np.square(x / 1)))),
                     ('classifier', LogisticRegression(random_state=0, solver='lbfgs'))])


#classifier = KNeighborsClassifier(n_neighbors=3)

classifier.fit(X_train, y_train)
print( 1 - classifier.score(X_test, y_test))
print( 1 - classifier.score(X_train, y_train))
print( 1 - cross_val_score(classifier, X_train, y_train, cv=10, scoring='accuracy').mean())
plt.scatter(X[:, 0], X[:, 1], c = y)
x1, x2 = np.mgrid[-5:5:0.01, -5:5:0.01]
y_hat = classifier.predict(np.c_[x1.ravel(), x2.ravel()]).reshape((x1.shape[0], x2.shape[0]))
plt.contour(x1, x2, y_hat, [0])


plt.show()