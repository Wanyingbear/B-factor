

import numpy as np

class WeightingCompute:
    def __init__(self, alpha=1.0, r=1.0):
        self.alpha = alpha
        self.r = r

    def compute_phi_matrix(self, X):
        """Compute the Phi matrix based on the input matrix X"""
        dist_matrix = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
        Phi = np.exp(-self.alpha * np.power(dist_matrix, self.r))
        return Phi

    def compute_inverse_row_sums(self, matrix):
        """Compute the inverse row sums of the input matrix"""
        if matrix.ndim != 2:
            raise ValueError("Input must be a 2D matrix.")

        rows, cols = matrix.shape
        if rows != cols:
            raise ValueError("Matrix must be square.")

        det = np.linalg.det(matrix)
        if det == 0:
            raise ValueError("Matrix is non-invertible (determinant is 0).")

        inverse_matrix = np.linalg.inv(matrix)

        row_sums = np.sum(inverse_matrix, axis=1)

        return row_sums

    def compute_weighting_vector(self, X):
        """Compute the weighting vector for the input matrix X"""
        Phi = self.compute_phi_matrix(X)
        
        weighting_vector = self.compute_inverse_row_sums(Phi)
        
        return weighting_vector
