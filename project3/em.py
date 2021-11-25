"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture

TINY = 1e-12

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    l_gauss = np.zeros((n, K))
    f = np.zeros((n, K))
    Xprime = np.ma.masked_array(X, mask=~(np.abs(X) > 0))
    d_entry = (np.abs(X) > 0).sum(axis=1)
    cost = 0
    for j in range(K):
        l_gauss[:, j] = -(d_entry / 2)*np.log(2 * np.pi * mixture.var[j])\
                        -(0.5 / mixture.var[j]) * np.square(np.linalg.norm((Xprime - mixture.mu[j]), axis=1))
        f[:, j] = np.log(mixture.p[j]+TINY) + l_gauss[:, j]
    for j in range(K):
        post[:, j] = np.exp(f[:, j] - logsumexp(f,axis=1))
        cost += np.sum((np.log(mixture.p[j]+TINY) + l_gauss[:, j] - np.log(post[:, j]+TINY)) * post[:, j])
    return (post, cost)

def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    mu = np.empty((K, d))
    var = np.empty(K)
    p = np.empty(K)
    Cu = np.abs(X) > 0
    Xprime = np.ma.masked_array(X, mask=~Cu)
    d_entry = Cu.sum(axis=1)
    for j in range(K):
        p[j] = (1 / n) * np.sum(post[:, j])
        mask_p = (Cu.T*post[:, j]).T
        mu[j] = np.sum(Xprime * mask_p,axis=0)/(mask_p.sum(axis=0))
        for l in range(d):
            if mask_p[:,l].sum() < 1: mu[j][l] = mixture.mu[j][l]
        var[j] = np.sum(np.square(np.linalg.norm((Xprime - mu[j]), axis=1))*post[:, j])
        var[j] /= np.sum(d_entry*post[:, j])
        var[j] = max(var[j], min_variance)
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
    error = 1e6
    cost = TINY
    iter = 0
    while (error > 1e-6 * np.abs(cost)):
        iter+=1
        prev_cost = cost
        (post, cost) = estep(X, mixture)
        mixture = mstep(X, post, mixture)
        error = np.abs(prev_cost - cost)
        if iter <= 2:error = 1e6
        print(f'iter:{iter},cost:{cost}')
    return mixture, post, cost

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    post, _ = estep(X,mixture)
    Xnew = np.copy(X)
    for i in range(Xnew.shape[0]):
        dim = (X[i] == 0)
        x_mu = np.zeros(Xnew.shape[1])
        for j in range(post.shape[1]):
            x_mu += post[i, j] * mixture.mu[j]
        Xnew[i][dim] = x_mu[dim]
    return Xnew