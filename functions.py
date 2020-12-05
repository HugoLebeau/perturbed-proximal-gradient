import numpy as np
import scipy.optimize as opt
from tqdm import tqdm

def prox(theta, gamma, g, t0=None, method='BFGS', grad_g=None):
    '''
    Proximal operator of gamma*g at point theta.

    Parameters
    ----------
    theta : float or ndarray, shape (n,)
        Point where to compute the proximal operator.
    gamma : float
        Scaling parameter.
    g : callable
        Function whose proximal operator is to be computed.
    t0 : float or ndarray, shape (n,), optional
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

def proximal_gradient(grad_f, g, theta0, gamma=1., niter=100, prox_g=None, method='BFGS', grad_g=None):
    '''
    Proximal gradient algorithm.

    Parameters
    ----------
    grad_f : callable
        Method for computing the gradient of f.
    g : callable
        Method for computing g.
    theta0 : float or ndarray, shape (n,)
        Initial guess.
    gamma : float or ndarray, shape (m,), optional
        Proximal step size. The default is 1..
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
    theta : ndarray, shape (m+1, d)
        Value of theta at each iteration.

    '''
    theta = np.zeros(niter+1) if np.isscalar(theta0) else np.zeros((niter+1, theta0.shape[0]))
    theta[0] = theta0
    if np.isscalar(gamma):
        gamma = np.ones(niter)*gamma
    if prox_g is None:
        prox_g = lambda theta, gamma: prox(theta, gamma, g, method=method, grad_g=grad_g).x
    for n in tqdm(range(niter)):
        theta[n+1] = prox_g(theta[n]-gamma[n]*grad_f(theta[n]), gamma[n])
    return theta
