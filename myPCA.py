import numpy as np
import pandas as pd
class myPCA():
    def __init__(self, n_components=2):
        ''' This function sets up the PCA algorithm '''
        self.n_components = n_components

    def first(self, iterable):
        return next(x for x in iterable)
    def fit(self, X):
        ''' This function fits the PCA algorithm '''
        # Checking if the n_componets is smaller then number of feature column in X
        if self.n_components > np.size(X, 1):
            print("Error | You cand transpose a dataset to a higher dimension")
            return

        # Getting the correlation matrix
        corr_mat = np.corrcoef(X.T)

        # Getting the eigenvectors and eigenvalues
        self.eig_vals, self.eig_vecs = np.linalg.eig(corr_mat)

        # Sorting the list of tuples (eigenvalue, eigenvector)
        self.eig_pairs = [(np.abs(self.eig_vals[i]), self.eig_vecs[:, i]) for i in range(len(self.eig_vals))]
        self.eig_pairs.sort(key=lambda x: x[0], reverse=True)

        # Calculating the explainet ration
        total = sum(self.eig_vals)
        self.explained_variance_ratio = [(i/total)* 100 for i in sorted(self.eig_vals, reverse= True)]
        self.cumulative_variance_ratio = np.cumsum(self.explained_variance_ratio)

        # Creating the projection matrix
        self.matrix_w = np.hstack(tuple(self.eig_pairs[i][1].reshape(np.size(X, 1)) for i in range(self.n_components)))

    def transform(self, X):
        ''' The data transformation to the new dimension space '''
        return X.dot(self.matrix_w)
