import numpy as np

class LinearSVM:
    def __init__(self, lr=0.01, lambda_param=0.01, epochs=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.epochs = epochs

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        self.W = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.epochs):
            for i, x_i in enumerate(X):
                condition = y_[i] * (np.dot(x_i, self.W) + self.b) >= 1
                if condition:
                    self.W -= self.lr * (2 * self.lambda_param * self.W)
                else:
                    self.W -= self.lr * (2 * self.lambda_param * self.W - np.dot(x_i, y_[i]))
                    self.b -= self.lr * y_[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.W) + self.b)
