DATA_FILE = input("Enter Data File: ")
epsilon =  float(input("Enter epsilon: "))

def inner(u, v):
    sum = 0
    for i in range(u.shape[0]):
        sum += u[i] * v[i]
    return sum

import math
def cos(u, v):
    u_norm = 0
    v_norm = 0
    for i in range(u.shape[0]):
        u_norm += u[i] * u[i]
        v_norm += v[i] * v[i]
    return inner(u, v) / math.sqrt(u_norm * v_norm)

def outer(u, v):
    outer_product = np.zeros((u.shape[0], v.shape[0]))
    for i in range(u.shape[0]):
        for j in range(v.shape[0]):
            outer_product[i, j] = u[i] * v[j]
    return outer_product

import numpy as np
X = np.loadtxt(DATA_FILE)

# A
mean = np.zeros(X.shape[1])
for col in range(X.shape[1]):
    sum = 0
    for row in range(X.shape[0]):
        sum += X[row, col]
    mean[col] = sum / X.shape[0]
    
print("Mean: ", mean)

centered = np.zeros(X.shape)
for col in range(X.shape[1]):
    for row in range(X.shape[0]):
        centered[row, col] = X[row, col] - mean[col]
        
variance = 0
for row in range(centered.shape[0]):
    variance += inner(centered[row, :], centered[row, :])
variance /= centered.shape[0]
print("Total Variance: ", variance)

# B
inner_covariance = np.zeros((centered.shape[1], centered.shape[1]))
for i in range(centered.shape[1]):
    for j in range(centered.shape[1]): 
        inner_covariance[i, j] = inner(centered[:, i], centered[:, j])
print("Inner Covariance ", inner_covariance)

outer_covariance = np.zeros((centered.shape[1], centered.shape[1]))
outer_products = [outer(centered[i, :], centered[i, :]) for i in range(centered.shape[0])]
for p in range(centered.shape[0]):
    for i in range(centered.shape[1]):
        for j in range(centered.shape[1]): 
            outer_covariance[i, j] += outer_products[p][i, j] / centered.shape[0]
print("Outer Covariance ", outer_covariance)

# C
cosines = np.zeros((centered.shape[1], centered.shape[1]))
for i in range(centered.shape[1]):
    for j in range(centered.shape[1]): 
        cosines[i, j] = cos(centered[:, i], centered[:, j])
print("Correlation ", cosines)
import matplotlib.pyplot as plt
plt.scatter(X[:, 1], X[:, 4])
plt.show()
plt.scatter(X[:, 1], X[:, 2])
plt.show()
plt.scatter(X[:, 2], X[:, 3])
plt.show()

#II
eigenvector = np.full((X.shape[1]), 1)
err = 1
while err > epsilon:
    temp = np.matmul(inner_covariance, eigenvector)
    temp = temp / np.max(temp)
    err = np.sum((temp - eigenvector)**2)
    eigenvector = temp
eigenvector /= np.sum(eigenvector**2)
print("Dominant Eigenvector: ", eigenvector)

proj = np.matmul(X, eigenvector)
print("Projection: ", proj)


