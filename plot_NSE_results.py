import itertools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.special import logsumexp

from utils import vec2mat

folder = 'NSE_outputs/'
solvers = ['Solver1', 'Solver2']
res = dict()
for file in os.listdir(folder):
    name = file.split('.')[0].split('_')[1]
    p = int(file.split('.')[0].split('_')[2][1:])
    if p not in res.keys():
        res[p] = dict()
    if name in solvers:
        res[p][name] = pd.read_csv(folder+file, index_col=0)
    else: # obs / theta
        res[p][name] = pd.read_csv(folder+file, header=None).values

m = dict()
for p in res.keys():
    m[p] = dict()
    niter2 = res[p]['Solver2'].shape[0]-1
    m[p]['Solver2'] = np.concatenate(([0], 500+(np.arange(1, niter2+1)**1.2).round().astype(int)))
    m[p]['Solver1'] = np.concatenate(([0], np.ones(int(np.round(m[p]['Solver2'].sum()/500)), dtype=int)*500))

fig, ax = plt.subplots(nrows=len(res.keys()), ncols=2, figsize=(15, 10))
for i, p in enumerate(sorted(res.keys())):
    for solver in solvers:
        cum_m = m[p][solver].cumsum()
        ax[i][0].plot(cum_m, res[p][solver]['rel_err'], label=solver)
        ax[i][1].plot(cum_m, res[p][solver]['F'], label=solver)
    ax[i][0].legend()
    ax[i][0].set_xlabel("MC samples")
    ax[i][0].set_ylabel("Relative error")
    ax[i][0].set_title(r"$p = {}$".format(p))
    ax[i][1].legend()
    ax[i][1].set_xlabel("MC samples")
    ax[i][1].set_ylabel("F statistic")
    ax[i][1].set_title(r"$p = {}$".format(p))
plt.show()

norm = Normalize(vmin=-4., vmax=4., clip=False)
fig, ax = plt.subplots(nrows=len(res.keys()), ncols=3, figsize=(15, 10))
for i, p in enumerate(sorted(res.keys())):
    for j, solver in enumerate(solvers):
        mat = vec2mat(res[p][solver].iloc[-1, :p*(p+1)//2].values)
        ax[i][j+1].imshow(mat, norm=norm, cmap='bwr', interpolation='none')
        ax[i][j+1].set_title(r"{}   p={}".format(solver, p))
    ax[i][0].imshow(res[p]['theta'], norm=norm, cmap='bwr', interpolation='none')
    ax[i][0].set_title(r"True $\theta$   $p={}$".format(p))
plt.show()

fig, ax = plt.subplots(nrows=len(res.keys()), ncols=1, figsize=(15, 10))
for i, p in enumerate(sorted(res.keys())):
    for solver in solvers:
        cum_m = m[p][solver].cumsum()
        ax[i].plot(cum_m, res[p][solver].iloc[:, :p*(p+1)//2].apply(np.linalg.norm, axis=1), label=solver)
    ax[i].legend()
    ax[i].set_xlabel("MC samples")
    ax[i].set_ylabel(r"$||\theta_n||$")
    ax[i].set_title(r"$p = {}$".format(p))
plt.show()

if 5 in res.keys():
    M, p, N = 5, 5, 250
    obs = res[p]['obs']
    niter1 = res[p]['Solver1'].shape[0]-1
    niter2 = res[p]['Solver2'].shape[0]-1
    lambda_reg1 = 2.5*np.sqrt(np.log(p)/N)
    lambda_reg2 = 2.5*np.sqrt(np.log(p)/N)
    Xp = np.array(list(itertools.product(range(M), repeat=p)))
    scal = lambda theta, x: np.sum(np.diag(theta)*x)+np.sum([theta[k, j]*np.float(x[k] == x[j]) for k in range(p) for j in range(k-1)])
    logZ = lambda theta: logsumexp([scal(theta, x) for x in Xp])
    ell = lambda theta: logsumexp([scal(theta, x) for x in obs])/obs.shape[0]-logZ(theta)
    g = lambda theta, lambda_reg: lambda_reg*np.sum([np.abs(theta[k, j]) for k in range(p) for j in range(k-1)])
    F = lambda theta, lambda_reg: -ell(theta)+g(theta, lambda_reg)
    F1 = np.array([F(vec2mat(res[p]['Solver1'].iloc[i, :p*(p+1)//2].values), lambda_reg1) for i in range(niter1+1)])
    F2 = np.array([F(vec2mat(res[p]['Solver2'].iloc[i, :p*(p+1)//2].values), lambda_reg2) for i in range(niter2+1)])
    valF = [F1, F2]
    for i, solver in enumerate(solvers):
        plt.plot(m[p][solver].cumsum(), valF[i], label=solver)
    plt.xlabel("MC Samples")
    plt.ylabel(r"$F(\theta_n)$")
    plt.title(r"$p={}$".format(p))
    plt.legend()
    plt.show()
