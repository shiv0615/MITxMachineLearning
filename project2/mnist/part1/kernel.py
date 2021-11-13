import math
import numpy as np
import scipy as scipy
### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    X_prime = np.matmul(X,Y.T)/c
    c_power_p = np.power(c,p)
    kernel = 0
    for i in range(p+1):
        coeff = math.factorial(p)/(math.factorial(p-i)*math.factorial(i))*c_power_p
        kernel += coeff*np.power(X_prime,i)
    return kernel

def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    n = X.shape[0]
    m = Y.shape[0]
    kernel_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            kernel_matrix[i, j] = np.exp(-gamma * np.sum(np.square(X[i, :] - Y[j, :])))
    # kernel_matrix = np.exp(-gamma*np.sum(np.square(X-Y),axis=1))
    return kernel_matrix