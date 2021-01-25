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


model = keras.Sequential()
model.add(layers.Dense(10, activation= 'tanh'))
model.add(layers.Dense(1))

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
model.compile('adam', 'mean_squared_error')
history = model.fit(X_train, y_train, validation_split = 1/6, epochs = 10000, verbose=2, callbacks = [es])

preds = model.predict(X_test)
preds[preds > 0] = 1
preds[preds < 0] = -1
from sklearn.metrics import accuracy_score
print(1-accuracy_score(y_test, preds))




import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c = y)
x1, x2 = np.mgrid[-5:5:0.01, -5:5:0.01]
y_hat = model.predict(np.c_[x1.ravel(), x2.ravel()]).reshape((x1.shape[0], x2.shape[0]))
plt.contour(x1, x2, y_hat, [0])
plt.show()



