import numpy as np
from numpy import random

class kmeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        m, n = X.shape
        # Randomly initialize centroids
        random_indices = random.choice(m, self.n_clusters, replace=False)
        print(random_indices)
        print(X[random_indices])
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            # Compute distances from data points to centroids
            print(X[:, np.newaxis])
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            # Assign clusters based on closest centroid
            labels = np.argmin(distances, axis=1)

            # Compute new centroids as mean of assigned points
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            # Check for convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

if __name__ == "__main__":
    X = random.randint(0, 100, (10, 2))
    kmeans_model = kmeans(n_clusters=3)
    kmeans_model.fit(X)
    labels = kmeans_model.predict(X)
    print("Centroids:\n", kmeans_model.centroids)
    print("Labels:\n", labels)
    print("Data points:\n", X)
    print("Data points with labels:\n", np.column_stack((X, labels)))