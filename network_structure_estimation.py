import argparse
import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm
from scipy.special import logsumexp

from utils import vec2mat, mat2vec
from prox_functions import prox_L1, stochastic_proximal_gradient

parser = argparse.ArgumentParser(description="Network structure estimation")
parser.add_argument('--p', type=int, metavar='P', help="Dimension.")
args = parser.parse_args()

np.random.seed(1)

M = 20 # number of possible states
p = args.p # dimension
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
theta_true_vec = mat2vec(theta_true)

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

def Gibbsf(theta, niter=100, x0=np.zeros(p, dtype=int), verbose=True):
    ''' Gibbs sampler to sample from f '''
    x = x0.copy()
    u = np.random.rand(niter*p)
    chain = np.zeros((niter, p), dtype=int)
    for it in tqdm(range(niter), disable=not verbose):
        for i in range(p):
            proba = np.exp(logcondf(x, i, theta))
            x[i] = sampling(proba, u[it*p+i])
        chain[it] = x
    return chain

# Generate observations
obs = Gibbsf(theta_true, niter=10+N)[10:]

# Useful variables
idv = mat2vec(np.eye(p, dtype=bool)) # indices of the diagonal of a vector
eps = 1e-8

# Parameters
def grad_f(theta, obs, x0, m=500, burn_in=10):
    z = Gibbsf(vec2mat(theta), niter=burn_in+m, x0=x0, verbose=False)[burn_in:]
    return mat2vec(np.mean([barB(xi) for xi in obs], axis=0)-np.mean([barB(zi) for zi in z], axis=0)), z[-1]
g = lambda theta_vec: np.sum(np.abs(theta_vec))
theta0 = np.zeros(p*(p+1)//2) # vector representation

# Solver 2
niter2 = 5*p
m2 = 500+(np.arange(1, niter2+1)**1.2).round().astype(int)
lambda_reg2 = 2.5*np.sqrt(np.log(p)/np.arange(1, niter2+1))
gamma2 = 25./(p*np.sqrt(50))
output2 = stochastic_proximal_gradient(grad_f, g, theta0, obs, m=m2, gamma=gamma2, lambda_=lambda_reg2, niter=niter2, prox_g=prox_L1)

df2 = pd.DataFrame(output2)
df2['rel_err'] = np.linalg.norm(output2-output2[-1], axis=1)/np.linalg.norm(output2[-1])
df2['sen'] = np.sum((np.abs(output2[:, ~idv]) > eps)*(np.abs(output2[-1, ~idv]) > eps), axis=1)/np.sum(np.abs(output2[:, ~idv]) > eps, axis=1)
df2['prec'] = np.sum((np.abs(output2[:, ~idv]) > eps)*(np.abs(output2[-1, ~idv]) > eps), axis=1)/np.sum(np.abs(output2[-1, ~idv]) > eps)
df2['F'] = 2*df2['sen']*df2['prec']/(df2['sen']+df2['prec'])
df2.to_csv("NSE_Solver2_p{}.csv".format(p))

# Solver 1
m1 = 500
niter1 = int(np.round(m2.sum()/m1))
lambda_reg1 = 2.5*np.sqrt(np.log(p)/np.arange(1, niter1+1))
gamma1 = 25./(p*np.arange(1, niter1+1)**0.7)
output1 = stochastic_proximal_gradient(grad_f, g, theta0, obs, m=m1, gamma=gamma1, lambda_=lambda_reg1, niter=niter1, prox_g=prox_L1)

df1 = pd.DataFrame(output1)
df1['rel_err'] = np.linalg.norm(output1-output1[-1], axis=1)/np.linalg.norm(output1[-1])
df1['sen'] = np.sum((np.abs(output1[:, ~idv]) > eps)*(np.abs(output1[-1, ~idv]) > eps), axis=1)/np.sum(np.abs(output1[:, ~idv]) > eps, axis=1)
df1['prec'] = np.sum((np.abs(output1[:, ~idv]) > eps)*(np.abs(output1[-1, ~idv]) > eps), axis=1)/np.sum(np.abs(output1[-1, ~idv]) > eps)
df1['F'] = 2*df1['sen']*df1['prec']/(df1['sen']+df1['prec'])
df1.to_csv("NSE_Solver1_p{}.csv".format(p))
