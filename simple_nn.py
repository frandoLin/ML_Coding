import numpy as np

class SimpleNeuralNet:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def fit(self, X, y):
        y_one_hot = np.eye(np.max(y) + 1)[y]

        for _ in range(self.epochs):
            z1 = X @ self.W1 + self.b1
            a1 = self.sigmoid(z1)
            z2 = a1 @ self.W2 + self.b2
            a2 = self.softmax(z2)

            dz2 = a2 - y_one_hot
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            dz1 = dz2 @ self.W2.T * self.sigmoid_deriv(a1)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

    def predict(self, X):
        a1 = self.sigmoid(X @ self.W1 + self.b1)
        a2 = self.softmax(a1 @ self.W2 + self.b2)
        return np.argmax(a2, axis=1)
