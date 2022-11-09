import numpy as np


class PCA():
    
    def __init__(self, n_components):
        
        self.W = None
        self.n_components = n_components
        
        
    def fit(self, x):
        
        cov_matrix = np.cov(x.T)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)
        argsort_eigen_vals = np.argsort(np.abs(eigen_vals))[::-1]
        self.W = eigen_vecs[:, argsort_eigen_vals[:self.n_components]]
    
    
    def transform(self, x):
        
        x_pca = x @ self.W
        
        return x_pca
    
    
    def fit_transform(self, x):
        
        cov_matrix = np.cov(x.T)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)
        argsort_eigen_vals = np.argsort(np.abs(eigen_vals))[::-1]
        self.W = eigen_vecs[:, argsort_eigen_vals[:self.n_components]]
        
        x_pca = x @ self.W
        
        return x_pca