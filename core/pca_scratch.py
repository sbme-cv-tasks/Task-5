import numpy as np

class PCA_Scratch:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        # 1. mean
        self.mean = np.mean(X, axis=0)

        # 2. center data
        X_centered = X - self.mean

        # 3. covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 4. eigenvalues + eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 5. sort descending
        idx = np.argsort(eigenvalues)[::-1]

        self.components = eigenvectors[:, idx[:self.n_components]]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)