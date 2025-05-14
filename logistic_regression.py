import numpy as np
from numpy import random

class LogisticRegression:
    def __init__ (self, learning_rate=0.01,):
        self.lr = learning_rate
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return 1/ (1 + np.exp(-z))
    
    def fit(self, x, y):
        m, n = x.shape
        self.w = np.zeros(n)
        self.b = 0
    
        y_pred = self.sigmoid(x@self.w + self.b)
        errors = y_pred - y
        dw = (1/m) * (x.T@errors)
        db = (1/m) * np.sum(errors)
        self.w -= self.lr * dw
        self.b -= self.lr * db

        return self.w, self.b
    
    def predict(self, x):
        y_pred = self.sigmoid(x@self.w + self.b)

        return np.where(y_pred > 0.5, 1, 0)
    
if __name__ == "__main__":
    lr = 1e-5
    print(lr)
    X = random.randint(0,100, (4,2))
    y = random.randint(0, 2, (4))

    model = LogisticRegression(lr)
    model.fit(X, y)
    y_pred = model.predict(X)
    print(y_pred)