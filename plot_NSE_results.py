import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folder = 'NSE_outputs/'
res = dict()
for file in os.listdir(folder):
    solver = file[4:11]
    p = int(file[13:-4])
    if p not in res.keys():
        res[p] = dict()
    res[p][solver] = pd.read_csv(folder+file)

m = dict()
for p in res.keys():
    m[p] = dict()
    m[p]['Solver2'] = 500+(np.arange(1, 5*p+1)**1.2).round().astype(int)
    m[p]['Solver1'] = np.ones(int(np.round(m[p]['Solver2'].sum()/500)), dtype=int)*500

fig, ax = plt.subplots(nrows=len(res.keys()), figsize=(10, 10))
for i, p in enumerate(res.keys()):
    for solver in res[p].keys():
        cum_m = np.concatenate(([0], m[p][solver].cumsum()))
        ax[i].plot(cum_m, res[p][solver]['rel_err'], label=solver)
    ax[i].legend()
    ax[i].set_xlabel("MC samples")
    ax[i].set_ylabel("Relative error")
plt.show()

fig, ax = plt.subplots(nrows=len(res.keys()), figsize=(10, 10))
for i, p in enumerate(res.keys()):
    for solver in res[p].keys():
        cum_m = np.concatenate(([0], m[p][solver].cumsum()))
        ax[i].plot(cum_m, res[p][solver]['F'], label=solver)
    ax[i].legend()
    ax[i].set_xlabel("MC samples")
    ax[i].set_ylabel("F statistic")
plt.show()
