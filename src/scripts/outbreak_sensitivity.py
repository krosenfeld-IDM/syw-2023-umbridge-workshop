"""
Sensitivity figures
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import pandas as pd
import numpy as np
import pyfcf

import paths
import methods

cols = ['C1', 'C2', 'C3', 'C4']
label_dict = dict(
    R0_variance = 'Spreader\nHeterogeneity',
    indiv_variance_acq = 'Catcher\nHeterogeneity',
    correlation_acq_trans = 'Spreader-Catcher\nCorrelation'
)
pyfcf.setup_matplotlib(font_size=12)

# Reference trajectory (Kermack-McKendric analytic solution)
def KMlimt (x,R0):
  return 1-x-np.exp(-x*R0)

# load data
db = methods.DataBrick(tpath = paths.emod / "model_covariance01" / "experiment_covariance01")
db_df = db.to_df()

# param_levels
param_keys = ['R0_variance', 'indiv_variance_acq', 'correlation_acq_trans']
param_levels = db_df[param_keys].drop_duplicates().iloc[::-1].reset_index(drop=True)

#####################################################
# Figure 1: Plot Z as a funtion of R0

# setup figure and axes
fc = pyfcf.FigConfig(1, 1, idx=3.8, idy=3.2, xm=[0.7, 0.08], ym=[0.6, 0.08])
fig = plt.figure(figsize=(fc.xs, fc.ys))
ax = plt.axes(fc.get_rect(0,0))

# add analytic solution
xref = np.linspace(1.01,2.0,200)
yref = np.zeros(xref.shape)
for k1 in range(yref.shape[0]):
  yref[k1] = spopt.brentq(KMlimt, 1e-5, 1, args=(xref[k1]))

ax.plot(np.concatenate((np.linspace(0.5,1.0,5), xref)),
        np.concatenate((np.zeros(5),yref)),
    '-',color=methods.DefaultColors().analytic, lw=5.0)

# add EMOD results
for col, (_, params) in zip(cols, param_levels.iterrows()):
    print(params)
    select_df = pd.merge(db_df, pd.DataFrame(params).T, on=param_keys)
    ax.plot(select_df['R0'], select_df['atk_frac'], 'o', color=col, ms=5)

# format axes
ax.set_xlim( 0.5, 1.75 )
ax.set_ylim(-0.01, 0.81)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(["{:0.0f}%".format(100*s) for s in ax.get_yticks()])
ax.set_xlabel('Reproductive Number')
ax.set_ylabel('Population Infected')
ax.set_ylim(-0.01, 0.81)


plt.savefig(paths.figures / "outbreak_sensitivyt_Z.png", transparent=1)

#####################################################
# Figure 2: Plot the samples

pyfcf.setup_matplotlib(font_size=8)

# setup figure and axes
ym = [0.4, 0.15, 0.06]
xm = [0.35, 0.15, 0.06]
idy = fc.idy/2 - ym[1]
idx = fc.idx/2 - xm[1]

fc = pyfcf.FigConfig(2, 2, idx=idx, idy=idy, xm=xm, ym=ym)
fig = plt.figure(figsize=(fc.xs, fc.ys))
print(fc.xs, fc.ys)
N = 1000

for ii, (col, (_, params)) in enumerate(zip(cols, param_levels.iterrows())):
    ix, iy = np.unravel_index(ii, (2,2))
    ax = plt.axes(fc.get_rect(ix, iy))
    # select_df = pd.merge(db_df, pd.DataFrame(params).T, on=param_keys)

    R0_VAR   = params['R0_variance']
    AQ_VAR   = params['indiv_variance_acq']
    rho      = params['correlation_acq_trans']

    R0_LN_SIG   = np.sqrt(np.log(R0_VAR+1.0))
    R0_LN_MU    = -0.5*R0_LN_SIG*R0_LN_SIG

    ACQ_LN_SIG  = np.sqrt(np.log(AQ_VAR+1.0))
    ACQ_LN_MU   = -0.5*ACQ_LN_SIG*ACQ_LN_SIG

    risk_vec   = np.random.lognormal(mean=ACQ_LN_MU, sigma=ACQ_LN_SIG, size=N)
    inf_vec    = np.random.lognormal(mean=R0_LN_MU,  sigma=R0_LN_SIG,  size=N)
    corr_vec   = inf_vec*(1 + rho*(risk_vec - 1))
        
    ax.plot(risk_vec,corr_vec,'1', markersize=3, color='k')

    ax.set_xlim(0,6)
    ax.set_ylim(0,6)

    for child in ax.get_children():
        if isinstance(child, mpl.spines.Spine):
            child.set_color(col)
            child.set_linewidth(3)    


    if ix > 0:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel('Spreading multiplier')
    if iy > 0:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Catching multiplier')
    

plt.savefig(paths.figures / "outbreak_sensitivy_grid.png", transparent=1)
