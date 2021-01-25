import sys

TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]

import numpy as np
X_train = np.loadtxt(TRAIN_FILE, delimiter = ',')
X_test = np.loadtxt(TEST_FILE, delimiter = ',')
y_train = np.copy(X_train[:, X_train.shape[1] - 1])
y_test = np.copy(X_test[:, X_test.shape[1] - 1])

X_train = np.c_[np.ones((X_train.shape[0],1)), X_train[:, 0:X_train.shape[1] - 1]] 
X_test = np.c_[np.ones((X_test.shape[0],1)), X_test[:, 0:X_test.shape[1] - 1]] 

Q = np.zeros(X_train.shape)
R = np.identity(Q.shape[1])


for i in range(Q.shape[1]):
    Q[:, i] = X_train[:, i]
    for j in range(i):
        proj = np.dot(X_train[:, i], Q[:, j]) / np.dot(Q[:, j], Q[:, j])
        Q[:, i] -= proj * Q[:, j]
        R[j, i] = proj

delta = np.sum(Q**2,axis=0)**(-1)
b = delta * np.matmul(Q.T, y_train)

w = np.zeros(b.shape)
for i in range(b.shape[0]):
    j = b.shape[0] - 1 - i
    row = R[j, :]
    w[j] = b[j] - np.dot(row, w)
    
print('weight vector: ', w)
print('L2-Norm: ', np.sqrt(np.sum(w**2)))

train_sse =  np.mean((y_train - np.matmul(X_train, w))**2)
train_tss = np.mean((y_train - np.mean(y_train))**2)
test_sse =  np.mean((y_test - np.matmul(X_test, w))**2)
test_tss = np.mean((y_test - np.mean(y_test))**2)
print('Train SSE: ', train_sse)
print('Train R^2: ', (train_tss - train_sse) / train_tss)
print('Test SSE: ', test_sse)
print('Test R^2: ', (test_tss - test_sse) / test_tss)