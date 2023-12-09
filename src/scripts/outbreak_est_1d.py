"""
Outbreak estimate given uncertainty in R0
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize     as spopt
import pyfcf

import paths
import methods

pyfcf.setup_matplotlib(font_size=10)

# Reference trajectory (Kermack-McKendric analytic solution)
def KMlimt (x,R0):
  return 1-x-np.exp(-x*R0)

def analytic_Z(R0):
    a_R0 = np.array(R0)
    res = []
    for R0_ in a_R0:
        if R0_ <= 1:
            res.append(0)
        else:
            res.append(spopt.brentq(KMlimt, 1e-5, 1, args=(R0_)))
    return res


def format_axes(ax):

    # format axes
    ax.set_yticklabels([])
    ax.set_xlim(None, None)
    # hide y axis and ticks
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # hide top axis
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')

    return ax

def plot_R0_dist(m,s,ax, bins=np.linspace(0.8,2.5), n=10000):
    """
    Plot a distribution of R0 values
    """

    # draw n samples from the distribution set by m and s
    R0 = np.random.randn(n)*s + m

    # plot the distribution
    ax.hist(R0, bins=bins, color=methods.DefaultColors().R0, histtype="step")

    # format axes
    ax = format_axes(ax)

    return ax

def plot_Z_dist(m, s, ax, emod_results=None, n=10000, bins=np.linspace(0,1)):
    """
    Plot a distribution of Z values
    """

    # draw n samples from the distribution set by m and s
    R0 = np.random.randn(n)*s + m
    Z = analytic_Z(R0)

    # plot the analytic distribution
    ax.hist(Z, bins=bins, histtype="stepfilled", color=methods.DefaultColors().analytic, 
        lw=1, density=True)

    # plot the EMOD distribution
    if emod_results is not None:
        ax.hist(emod_results["atk_frac"], bins=bins, histtype="step", color=methods.DefaultColors().emod, 
            lw=1, density=True)

    # format axes
    ax = format_axes(ax)

    return ax

###################################################


R0_ranges = [[1.3, 1.8], [1.4, 2.2], [1.5, 2], [1.7, 1.8], [1.5, 2]]
names = ['Flu', 'Mumps', 'Smallpox', 'Diptheria', 'Rubella']

# read in EMOD results
emod_results = methods.DataBrick(tpath = paths.emod / "model_covariance01" / "experiment_workshop_1d_00").to_df()

# setup figure
ny = len(R0_ranges)
nx = 2
idx = 2.0
idy = 0.75
xm = [0.05, 0.15, 0.12]
ym = [0.55] + (ny-1)*[0.08] + [0.1]

fc = pyfcf.FigConfig(nx, ny, idx=idx, idy=idy, xm=xm, ym=ym)
fig = plt.figure(figsize=(fc.xs, fc.ys))

# loop through axes and plot R0 distributions:
for iy in range(len(R0_ranges)):
    # get axes R0 range and name
    R0_range = R0_ranges[iy]
    name = names[iy]
    # create axes
    ax = plt.axes(fc.get_rect(0,iy))
    # get mean and std
    m = np.mean(R0_range)
    s = (R0_range[1]-R0_range[0])/3
    # plot
    plot_R0_dist(m,s,ax)
    ax.text(ax.get_xlim()[0], ax.get_ylim()[0], name, ha="left", va="bottom", fontsize=10)
    # add to plot
    if iy == 0:
        ax.set_xlabel("Reproductive Number")
    else: 
        ax.set_xticklabels([])

# loop through axes and plot analytic+EMOD solutions
for iy in range(len(R0_ranges)):
    # get axes R0 range and name
    R0_range = R0_ranges[iy]
    name = names[iy]
    # create axes
    ax = plt.axes(fc.get_rect(1,iy))
    # get mean and std
    m = np.mean(R0_range)
    s = (R0_range[1]-R0_range[0])/3
    # get EMOD results
    select_emod_results = emod_results[emod_results["R0_name"] == name]
    # plot
    plot_Z_dist(m,s,ax,select_emod_results)
    # add to plot
    xlim = ax.get_xlim()
    if iy == 0:
        ax.set_xlabel("Fraction Infected")
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels(["{:0.0f}%".format(100*s) for s in xticks])
    else: 
        ax.set_xticklabels([])
    ax.set_xlim(xlim)

plt.savefig(paths.figures / 'outbreak_est_1d_R0.png', transparent=0)

    