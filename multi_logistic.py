import numpy as np
from numpy import random

class LogisticRegression:
    def __init__(self, learning_rate=1e-3):
        self.lr = learning_rate
        self.w = None
        self.b = None
        self.classes = None

    def softmax(self, z):
        exp_z = np.exp(z-np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, x, y):
        m, n = x.shape
        if not self.classes:
            self.classes = np.unique(y)
        k = len(self.classes)

        y_one_hot = np.zeros((m,k))
        for i in range(m):
            y_one_hot[i, np.where(self.classes == y[i])[0][0]] = 1

        self.w = np.zeros((n,k))
        self.b = np.zeros(k)

        y_pred = self.softmax(x@self.w + self.b)

        error = y_pred - y_one_hot
        dw = (1/m) * (x.T@error)
        db = (1/m) * np.sum(error, axis=0)

        self.w -= self.lr * dw
        self.b -= self.lr * db

        return self.w, self.b
    
    def predict(self, x):
        y_pred = self.softmax(x@self.w + self.b)
        
        return self.classes[np.argmax(y_pred, axis=1)]

if __name__ == "__main__":
    lr = 1e-3

    X = random.randint(0, 100, (20, 2))
    y = random.randint(0,3, (20))

    model = LogisticRegression(lr)
    model.fit(X, y)
    y_pred = model.predict(X)

    # Print results
    print("Predictions:", y_pred)
    print("Actual:     ", y)
    print("Accuracy:   ", np.mean(y_pred == y))



