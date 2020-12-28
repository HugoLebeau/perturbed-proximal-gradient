import numpy as np
import scipy.optimize as opt
from tqdm import tqdm

def prox_L1(theta, gamma):
    '''
    Proximal operator of the L1 norm.

    Parameters
    ----------
    theta : ndarray, shape(d,)
        Point where to compute the proximal operator.
    gamma : float
        Scaling parameter.

    Returns
    -------
    ndarray, shape(d,)
        The result of the proximal operator.

    '''
    return np.sign(theta)*np.maximum(np.abs(theta)-gamma, 0.)

def prox(theta, gamma, g, t0=None, method='BFGS', grad_g=None):
    '''
    Proximal operator of gamma*g at point theta.

    Parameters
    ----------
    theta : float or ndarray, shape (d,)
        Point where to compute the proximal operator.
    gamma : float
        Scaling parameter.
    g : callable
        Function whose proximal operator is to be computed.
    t0 : float or ndarray, shape (d,), optional
        Initial guess. If None, set to theta. The default is None.
    method : str or callable, optional
        Type of solver (see scipy.optimize.minimize). The default is 'BFGS'.
    grad_g : callable, optional
        Method for computing the gradient of g. If None, numerically computed.
        The default is None.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The optimization results.

    '''
    fun = lambda t: g(t)+np.dot(t-theta, t-theta)/(2.*gamma)
    jac = None if grad_g is None else (lambda t: grad_g(t)+(t-theta)/gamma)
    if t0 is None:
        t0 = theta
    return opt.minimize(fun, t0, method=method, jac=jac)

def proximal_gradient(grad_f, g, theta0, gamma=1., lambda_=1., niter=100, prox_g=None, method='BFGS', grad_g=None):
    '''
    Proximal gradient algorithm.

    Parameters
    ----------
    grad_f : callable
        Method for computing the gradient of f.
    g : callable
        Method for computing g.
    theta0 : float or ndarray, shape (d,)
        Initial guess.
    gamma : float or ndarray, shape (n,), optional
        Proximal step size. The default is 1..
    lambda_ : float or ndarray, shape (n,), optional
        Multiplicating factor applied to g.
    niter : int, optional
        Number of iterations. The default is 100.
    prox_g : callable, optional
        Proximal operator of g. If None, numerically computed. The default is
        None.
    method : str or callable, optional
        Type of solver to numerically compute the proximal operator of g (see
        scipy.optimize.minimize).
        The default is 'BFGS'.
    grad_g : callable, optional
        Method for computing the gradient of g (in case prox_g not given). The
        default is None.

    Returns
    -------
    theta : ndarray, shape (n+1, d)
        Value of theta at each iteration.

    '''
    theta = np.zeros(niter+1) if np.isscalar(theta0) else np.zeros((niter+1, theta0.shape[0]))
    theta[0] = theta0
    if np.isscalar(gamma):
        gamma = np.ones(niter)*gamma
    if np.isscalar(lambda_):
        lambda_ = np.ones(niter)*lambda_
    if prox_g is None:
        prox_g = lambda theta, gamma: prox(theta, gamma, g, method=method, grad_g=grad_g).x
    for n in tqdm(range(niter)):
        theta[n+1] = prox_g(theta[n]-gamma[n]*grad_f(theta[n]), lambda_[n]*gamma[n])
    return theta

def stochastic_proximal_gradient(grad_f, x0, g, theta0, obs, m=500, gamma=1., lambda_=1., niter=100, prox_g=None, method='BFGS', grad_g=None):
    '''
    Perturbed proximal gradient algorithm.

    Parameters
    ----------
    grad_f : callable
        Method for estimating with MCMC the gradient of f at a point, given
        observations and a batch size. It should also return the last sample of
        the Markov chain.
    x0 : ndarray
        Starting point of the MCMC.
    g : callable
        Method for computing g.
    theta0 : float or ndarray, shape (d,)
        Initial guess.
    obs : ndarray
        Observations used for the estimation of the gradient of f.
    m : int or ndarray, shape (n,)
        Batch size for the estimation of the gradient of f.
    gamma : float or ndarray, shape (mn optional
        Proximal step size. The default is 1..
    lambda_ : float or ndarray, shape (n,), optional
        Multiplicating factor applied to g.
    niter : int, optional
        Number of iterations. The default is 100.
    prox_g : callable, optional
        Proximal operator of g. If None, numerically computed. The default is
        None.
    method : str or callable, optional
        Type of solver to numerically compute the proximal operator of g (see
        scipy.optimize.minimize).
        The default is 'BFGS'.
    grad_g : callable, optional
        Method for computing the gradient of g (in case prox_g not given). The
        default is None.

    Returns
    -------
    theta : ndarray, shape (n+1, d)
        Value of theta at each iteration.

    '''
    theta = np.zeros(niter+1) if np.isscalar(theta0) else np.zeros((niter+1, theta0.shape[0]))
    theta[0] = theta0
    x = x0.copy()
    if np.isscalar(m):
        m = np.ones(niter, dtype=int)*m
    if np.isscalar(gamma):
        gamma = np.ones(niter)*gamma
    if np.isscalar(lambda_):
        lambda_ = np.ones(niter)*lambda_
    if prox_g is None:
        prox_g = lambda theta, gamma: prox(theta, gamma, g, method=method, grad_g=grad_g).x
    for n in tqdm(range(niter)):
        nabla_f, x = grad_f(theta[n], obs, x, m[n])
        theta[n+1] = prox_g(theta[n]-gamma[n]*nabla_f, lambda_[n]*gamma[n])
    return theta
