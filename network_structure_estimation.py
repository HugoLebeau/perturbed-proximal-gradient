import numpy as np
from utils import vec2mat
from prox_functions import prox_L1, proximal_gradient

M = 20 # number of possible states
p = 50 # dimension

# Generation of the true theta
theta = np.zeros((p, p))
nb_nonzero = np.int(np.round(np.random.randn()*(p/4.)+p)) # number of non-zero elements below the diagonal
lines = np.random.randint(1, p, nb_nonzero)
for i in lines:
    j = np.random.randint(0, i)
    u = np.random.rand()
    sgn = np.random.randint(0, 2)*2-1
    theta[i, j] = sgn*(3.*u+1.) # uniform on [-4, -1] U [1, 4]
    theta[j, i] = theta[i, j]

B0 = lambda x: x
B = lambda x, y: float(x == y) if (np.isscalar(x) and np.isscalar(y)) else (x == y).astype(float)

def barB(x):
    ''' Matrix of B(x_i, x_j) (B0(x_i) on the diagonal) '''
    mat = np.diag(B0(x))
    for k in range(p):
        elems = B(x[k], x[k+1:])
        mat[k, k+1:] = elems
        mat[k+1:, k] = elems
    return mat

def logf(x):
    ''' Returns log f(x) + C '''
    mat = np.diag(np.diag(B0(x)))
    for k in range(p):
        mat[k+1:, k] = B(x[k], x[k+1:])
    return np.sum(theta*mat)

grad_f ? # implement a Gibbs sampler

g = lambda theta_vec: np.sum(np.abs(theta_vec))
theta0 = np.zeros(p*(p+1)//2) # vector representation
niter = 5*p
gamma = 25.*/(p*np.arange(1, niter+1)**0.7)
lambda_reg = 2.5*np.sqrt(np.log(p)/np.arange(1, niter+1))

output = proximal_gradient(grad_f, g, theta0, gamma=1., lambda_=lambda_reg, niter=niter, prox_g=prox_L1)
theta_est = vec2mat(output)
