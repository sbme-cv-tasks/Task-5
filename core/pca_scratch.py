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
    

# import numpy as np

# class PCA_FromScratch:
#     def __init__(self, n_components=10, n_iter=100):
#         self.n_components = n_components
#         self.n_iter = n_iter
#         self.mean = None
#         self.components = None
#         self.eigenvalues = None

#     def _power_iteration(self, A):
#         n = A.shape[0]
#         b = np.random.rand(n)
#         b = b / np.linalg.norm(b)

#         for _ in range(self.n_iter):
#             b = A @ b
#             norm = np.linalg.norm(b)
#             if norm == 0:
#                 break
#             b = b / norm

#         eigenvalue = b.T @ A @ b
#         eigenvector = b
#         return eigenvalue, eigenvector

#     def fit(self, X):
#         # 1) mean center
#         self.mean = np.mean(X, axis=0)
#         Xc = X - self.mean

#         # 2) covariance
#         C = (Xc.T @ Xc) / (Xc.shape[0] - 1)

#         components = []
#         eigenvalues = []

#         A = C.copy()

#         # 3) extract top k eigenvectors
#         for _ in range(self.n_components):
#             val, vec = self._power_iteration(A)

#             components.append(vec)
#             eigenvalues.append(val)

#             # 4) deflation
#             A = A - val * np.outer(vec, vec)

#         self.components = np.array(components).T   # columns
#         self.eigenvalues = np.array(eigenvalues)

#     def transform(self, X):
#         Xc = X - self.mean
#         return Xc @ self.components

#     def fit_transform(self, X):
#         self.fit(X)
#         return self.transform(X)