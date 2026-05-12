import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

class PCADMD:
    def __init__(self, latent_dim=8):
        self.latent_dim = latent_dim
        self.pca = PCA(n_components=latent_dim)
        self.model = LinearRegression(fit_intercept=False)
        self.K = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None

    def fit(self, X, X_next):
        Z = self.pca.fit_transform(X)
        Z_next = self.pca.transform(X_next)

        self.model.fit(Z, Z_next)
        self.K = self.model.coef_

        eigvals, eigvecs = np.linalg.eig(self.K)
        idx = np.argsort(-np.abs(eigvals))

        self.eigenvalues_ = eigvals[idx]
        self.eigenvectors_ = eigvecs[:, idx]
        return self

    def predict(self, X):
        Z = self.pca.transform(X)
        Z_pred = Z @ self.K.T
        return self.pca.inverse_transform(Z_pred)
