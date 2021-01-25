import sys
TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]

HIDDEN_WIDTH = int(sys.argv[3])
LEARNING_RATE = float(sys.argv[4])
EPOCHS = int(sys.argv[5])


import numpy as np
X_train = np.genfromtxt(TRAIN_FILE, delimiter = ',')
X_test = np.genfromtxt(TEST_FILE, delimiter = ',')

INPUT_WIDTH = X_train.shape[1]

y_train = X_train[:, INPUT_WIDTH - 1]
y_test = X_test[:, INPUT_WIDTH - 1]
X_train = np.delete(X_train, INPUT_WIDTH - 1, axis=1)
X_test = np.delete(X_test, INPUT_WIDTH - 1, axis=1)

X_test = (X_test - np.min(X_train, axis = 0)) / (np.max(X_train, axis = 0) - np.min(X_train, axis = 0))
X_train = (X_train - np.min(X_train, axis = 0)) / (np.max(X_train, axis = 0) - np.min(X_train, axis = 0))

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

OUTPUT_WIDTH = int(np.max(y_train))
Y_train = np.zeros((X_train.shape[0], OUTPUT_WIDTH))
for i in range(OUTPUT_WIDTH):
    Y_train[np.where(y_train == i + 1),  i] = 1 

b_h = np.random.rand(HIDDEN_WIDTH)
b_o = np.random.rand(OUTPUT_WIDTH)
W_h = np.random.rand(INPUT_WIDTH, HIDDEN_WIDTH)
W_o = np.random.rand(HIDDEN_WIDTH, OUTPUT_WIDTH)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for _ in range(EPOCHS):
    for i in range(X_train.shape[0]):
        x = X_train[i, :]
        y = Y_train[i, :]
        z = sigmoid(b_h + np.matmul(W_h.T, x))
        o = sigmoid(b_o + np.matmul(W_o.T, z))
        delta_o = o * (1 - o) * (o - y)
        delta_h = z * (1 - z) * np.matmul(W_o, delta_o)
        b_o = b_o - LEARNING_RATE * delta_o
        b_h = b_h - LEARNING_RATE * delta_h
        W_o = W_o - LEARNING_RATE * np.outer(z, delta_o)
        W_h = W_h - LEARNING_RATE * np.outer(x, delta_h)

matches = 0
for i in range(X_test.shape[0]):
    x = X_test[i, :]
    y = y_test[i]
    y_pred = sigmoid(b_o + np.matmul(W_o.T, sigmoid(b_h + np.matmul(W_h.T, x))))
    matches += (y == np.argmax(y_pred) + 1)
    
print(matches/ X_test.shape[0])
print('b_h', b_h)
print('b_o', b_o)   
print('W_h', W_h)
print('W_o', W_o) 
    