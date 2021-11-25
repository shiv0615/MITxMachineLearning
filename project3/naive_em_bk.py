"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    fac = 1 / (2 * np.pi * mixture.var) ** (d / 2)
    post = np.zeros((n, K))
    cost = 0
    for i in range(n):
        gauss_sum = 0
        for j in range(K):
            z = (X[i, :] - mixture.mu[j])
            post[i, j] = np.multiply(fac[j] * np.exp(-(0.5 / mixture.var[j]) * np.square(np.linalg.norm(z))),
                                     mixture.p[j])
            gauss_sum += post[i, j]
        post[i, :] /= gauss_sum
        cost += np.sum(post[i, :] * np.log(gauss_sum))
    return (post, cost)


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    mu = np.empty((K, d))
    var = np.empty(K)
    p = np.empty(K)
    for j in range(K):
        p[j] = (1 / n) * np.sum(post[:, j])
        mu[j] = np.sum((X.T * post[:, j]), axis=1) / (n * p[j])
        var[j] = (1 / (d * n * p[j])) * (np.sum(post[:, j] * np.square(np.linalg.norm((X - mu[j]), axis=1))))
    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_cost = None
    cost = None
    while (prev_cost is None or np.abs(prev_cost - cost) > 1e-6 * np.abs(cost)):
        prev_cost = cost
        (post, cost) = estep(X, mixture)
        mixture = mstep(X, post)
    return mixture, post, cost
