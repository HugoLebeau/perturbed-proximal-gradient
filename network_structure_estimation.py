import numpy as np
from numba import njit
from tqdm import tqdm
from scipy.special import logsumexp

from utils import vec2mat
from prox_functions import prox_L1, proximal_gradient

np.random.seed(1)

M = 20 # number of possible states
p = 50 # dimension
N = 250 # sample size

# Generation of the true theta
theta_true = np.zeros((p, p))
nb_nonzero = np.int(np.round(np.random.randn()*(p/4.)+p)) # number of non-zero elements below the diagonal
lines = np.random.randint(1, p, nb_nonzero)
for i in lines:
    j = np.random.randint(0, i)
    u = np.random.rand()
    sgn = np.random.randint(0, 2)*2-1
    theta_true[i, j] = sgn*(3.*u+1.) # uniform on [-4, -1] U [1, 4]
    theta_true[j, i] = theta_true[i, j]

@njit
def B0(x):
    return x

@njit
def B(x, y):
    return np.where(x == y, 1, 0)

@njit
def barB(x):
    ''' Matrix of B(x_i, x_j) (B0(x_i) on the diagonal) '''
    mat = np.diag(B0(x))
    for k in range(p):
        elems = B(x[k], x[k+1:])
        mat[k, k+1:] = elems
        mat[k+1:, k] = elems
    return mat

@njit
def logf(x, theta):
    ''' log f(x) + log Z '''
    mat = np.diag(B0(x))
    for k in range(p):
        mat[k+1:, k] = B(x[k], x[k+1:])
    return np.sum(theta*mat)

def logcondf(x, i, theta):
    ''' Compute p(x_i|x_-i) for all x_i '''
    all_x = np.tile(x, (M, 1))
    all_x[:, i] = np.arange(M, dtype=int)
    all_logfx = np.array([logf(x, theta) for x in all_x])
    return all_logfx-logsumexp(all_logfx)

@njit
def sampling(proba, u=np.random.rand()):
    ''' Sample from {0, ..., M-1} with the given distribution '''
    s, k = proba[0], 0
    while s < u and k < M:
        k += 1
        s += proba[k]
    return k

def Gibbsf(theta, niter=100, x0=np.zeros(p, dtype=int)):
    ''' Gibbs sampler to sample from f '''
    x = x0.copy()
    u = np.random.rand(niter*p)
    chain = np.zeros((niter, p), dtype=int)
    for it in tqdm(range(niter)):
        for i in range(p):
            proba = np.exp(logcondf(x, i, theta))
            x[i] = sampling(proba, u[it*p+i])
        chain[it] = x
    return chain

# Generation of observations
obs = Gibbsf(theta_true, niter=10+N)[10:]

def grad_f(theta, obs, m=500, burn_in=10):
    z = Gibbsf(theta, niter=burn_in+m)[burn_in:]
    return np.mean([barB(xi) for xi in obs], axis=0)-np.mean([barB(zi) for zi in z], axis=0)

# g = lambda theta_vec: np.sum(np.abs(theta_vec))
# theta0 = np.zeros(p*(p+1)//2) # vector representation
# niter = 5*p
# gamma = 25./(p*np.arange(1, niter+1)**0.7)
# lambda_reg = 2.5*np.sqrt(np.log(p)/np.arange(1, niter+1))

# output = proximal_gradient(grad_f, g, theta0, gamma=1., lambda_=lambda_reg, niter=niter, prox_g=prox_L1)
# theta_est = vec2mat(output)
