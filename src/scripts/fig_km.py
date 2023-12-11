"""
Base figure showing the Kermack-McKendric analystiz solution for attack fraction
given R0
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt

import pyfcf
import paths
import methods

pyfcf.setup_matplotlib(font_size=12)

# Reference trajectory (Kermack-McKendric analytic solution)
def KMlimt (x,R0):
  return 1-x-np.exp(-x*R0)

# setup figure
fc = pyfcf.FigConfig(1, 1, idx=4.15, idy=2.98, 
    xm=[0.8, 0.05], ym=[0.55, 0.08])
fig = plt.figure(figsize=(fc.xs, fc.ys))
ax = plt.axes(fc.get_rect(0,0))

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
ax.set_xlabel('Reproductive Number')
ax.set_ylabel('Population Infected')
ax.set_ylim(-0.01, 0.81)

plt.savefig(paths.figures / 'KMlimit.png', transparent=1)