import numpy as np
import copy


class PCA():
    """
    PCA. A class to reduce dimensions
    """

    def __init__(self, retain_ratio):
        """

        :param retain_ratio: percentage of the variance we maitain (see slide for definition)
        """
        self.retain_ratio = retain_ratio

    @staticmethod
    def mean(x):
        """
        returns mean of x
        :param x: matrix of shape (n, m)
        :return: mean of x of with shape (m,)
        """
        return x.mean(axis=0)

    @staticmethod
    def cov(x):
        """
        returns the covariance of x,
        :param x: input data of dim (n, m)
        :return: the covariance matrix of (m, m)
        """
        return np.cov(x.T)

    @staticmethod
    def eig(c):
        """
        returns the eigval and eigvec
        :param c: input matrix of dim (m, m)
        :return:
            eigval: a numpy vector of (m,)
            eigvec: a matrix of (m, m), column ``eigvec[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``eigval[i]``
            Note: eigval is not necessarily ordered
        """

        eigval, eigvec = np.linalg.eig(c)
        eigval = np.real(eigval)
        eigvec = np.real(eigvec)
        return eigval, eigvec


    def fit(self, x):
        """
        fits the data x into the PCA. It results in self.eig_vecs and self.eig_values which will
        be used in the transform method
        :param x: input data of shape (n, m); n instances and m features
        :return:
            sets proper values for self.eig_vecs and eig_values
        """

        self.eig_vals = None
        self.eig_vecs = None

        x = x - PCA.mean(x)
        ########################################
        #       YOUR CODE GOES HERE            #
        ########################################

        # Get the covariance matrix
        cov_mat = PCA.cov(x)
        # Get the eigen valus and vectors
        eigval, eigvec = PCA.eig(cov_mat)

        # Sort the idencices of the eigvalues
        idx = np.argsort(eigval)
        # Wrong way, want descending
        eigidx = idx[::-1]

        # Get total variance
        total_var = sum(eigval)
        k_variance = 0

        # Figure out how many eigenvalues we want
        i = 0
        while k_variance < self.retain_ratio * total_var:
            k_variance += eigval[eigidx[i]]
            i += 1

        # Set the eigen values
        self.eig_vals = np.array([eigval[eigidx[j]] for j in range(i)])
        # Mess with columns and vectors, then set eigen vectors
        self.eig_vecs = np.ndarray((561, eigvec.shape[1]))
        for j in range(i):
            self.eig_vecs[:,j] = eigvec[:,eigidx[j]]

    def transform(self, x):
        """
        projects x into lower dimension based on current eig_vals and eig_vecs
        :param x: input data of shape (n, m)
        :return: projected data with shape (n, len of eig_vals)
        """

        if isinstance(x, np.ndarray):
            x = np.asarray(x)
        if self.eig_vecs is not None:
            return np.matmul(x, self.eig_vecs)
        else:
            return x
