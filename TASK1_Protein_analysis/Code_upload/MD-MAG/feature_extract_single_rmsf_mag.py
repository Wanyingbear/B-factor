# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:05:22 2024

@author: Wanying
"""

import numpy as np

class WeightingCompute:
    def __init__(self, alpha=1.0, r=1.0):
        
        self.alpha = alpha
        self.r = r

    def compute_phi_matrix(self, X):
        # Compute distance matrix (Euclidean norm)
        dist_matrix = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
        # Compute generalized exponential matrix e^{-alpha * d^r}
        Phi = np.exp(-self.alpha * np.power(dist_matrix, self.r))
        return Phi

    def compute_inverse_row_sums(self, matrix):
        # Check if the matrix is 2D
        if matrix.ndim != 2:
            raise ValueError("Input must be a 2D matrix.")

        # Check if the matrix is square
        rows, cols = matrix.shape
        if rows != cols:
            raise ValueError("Matrix must be square.")

        # Check if the matrix is invertible (via determinant)
        det = np.linalg.det(matrix)
        if det == 0:
            raise ValueError("Matrix is not invertible (determinant is 0).")

        # Compute the inverse matrix
        inverse_matrix = np.linalg.inv(matrix)

        # Compute the row sums of the inverse matrix
        row_sums = np.sum(inverse_matrix, axis=1)

        return row_sums

    def compute_weighting_vector(self, X):
        
        # Compute the Phi matrix
        Phi = self.compute_phi_matrix(X)

        # Compute the row sum vector of the inverse Phi matrix
        weighting_vector = self.compute_inverse_row_sums(Phi)

        # Return the weighting vector (as a numpy array)
        return weighting_vector
