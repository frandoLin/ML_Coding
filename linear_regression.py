import numpy as np
from numpy import random
# W = np.array([[1, 2], [2,3]])
# print(W)
# print(W.shape)

# X = np.array([[1, 2], [3, 4], [5, 6]])
# print(X)

# print(X.shape)
# print(np.dot(X, W))
# print(np.dot(X, W).shape)
lr = 1e5
print(lr)
W = random.randint(0, 10, (2))
X = random.randint(0,100, (4,2))
y = random.randint(0, 1, (4))
b = 0

y_pred = X@W + b
error = y_pred - y

# L = sum(error**2) / X.shape[0]

dw = (2/X.shape[0]) * (X.T@error)
db = (2/X.shape[0]) * error
print(dw.shape)
print(W.shape)
W -= lr*dw
b -= lr*db




