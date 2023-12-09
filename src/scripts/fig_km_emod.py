"""
2 figures showing basic EMOD simulation results and KM example
"""
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as spopt

import pyfcf

import paths
import methods

pyfcf.setup_matplotlib(font_size=12)

DIRNAME = 'model_covariance01/experiment_workshop00'
results = methods.DataBrick(paths.emod / DIRNAME).to_df()
select_res = results[results["daily_import_pressure"] == 1].set_index("run_number")
print(len(select_res))
######################################################
# Figure 1: example trace
example_trace = select_res.loc[48]
fc = pyfcf.FigConfig(1, 1, idx=4.15, idy=2.98, 
    xm=[0.8, 0.05], ym=[0.55, 0.05])
fig = plt.figure(figsize=(fc.xs, fc.ys))
ax = plt.axes(fc.get_rect(0,0))
ax.plot(example_trace["timeseries"], "-", color=methods.DefaultColors().emod)
ax.set_ylim(0, None)
ax.set_xlim(0, None)
ax.set_xlabel("Days")
ax.set_ylabel("New Infections")
plt.savefig(paths.figures / "example_trace.png", transparent=True)

######################################################
# Figure 2 (build): Simulation results reproducing KM solution

def fig2_build():
  fc = pyfcf.FigConfig(1, 1, idx=4.15, idy=2.98, 
      xm=[0.8, 0.05], ym=[0.55, 0.08])
  fig = plt.figure(figsize=(fc.xs, fc.ys))
  ax = plt.axes(fc.get_rect(0,0))

  # Reference trajectory (Kermack-McKendric analytic solution)
  def KMlimt (x,R0):
    return 1-x-np.exp(-x*R0)

  xref = np.linspace(1.01,2.0,200)
  yref = np.zeros(xref.shape)
  for k1 in range(yref.shape[0]):
    yref[k1] = spopt.brentq(KMlimt, 1e-5, 1, args=(xref[k1]))

  ax.plot(np.concatenate((np.linspace(0.5,1.0,5), xref)),
          np.concatenate((np.zeros(5),yref)),
      '-',color=methods.DefaultColors().analytic, lw=5.0,label='Attack Rate - Analytic Solution')

  ax.set_xlim( 0.5, 1.75 )
  ax.set_ylim(-0.01, 0.81)
  ax.set_yticks(ax.get_yticks())
  ax.set_yticklabels(["{:0.0f}%".format(100*s) for s in ax.get_yticks()])
  ax.set_ylim(-0.01, 0.81)
  ax.set_ylabel('Population Infected')
  return ax

ax = fig2_build()
ax.plot(example_trace["R0"], example_trace["atk_frac"], 'o', color=methods.DefaultColors().emod)  
plt.savefig(paths.figures / 'EMOD_reproduces_KMlimit_1.png', transparent=1)

ax = fig2_build()
ax.plot(select_res["R0"].values, select_res["atk_frac"].values, 'o', color=methods.DefaultColors().emod, alpha=0.6)
plt.savefig(paths.figures / 'EMOD_reproduces_KMlimit_2.png', transparent=1)


