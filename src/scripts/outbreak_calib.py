"""
Plot example example
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import corner
import pyfcf

import paths
import methods

# setup matplotlib settings
pyfcf.setup_matplotlib(font_size=12)

# calibration directory
tpath = paths.emod / "model_calibration01" / "experiment_workshop_calib05"

# load results
emod_results = methods.CalibBrick(tpath=tpath)

# which vars were included in the calibration? 
# calib_vars = ['R0_variance', 'indiv_variance_acq', 'correlation_acq_trans']
calib_vars = list(emod_results.param_calib["EXP_OPTIMIZE"].keys())
ndim = len(calib_vars)

# select what to plot
df_emod_results = emod_results.to_df()

for ii, iter_thresh in enumerate([0, 1, 4, df_emod_results["iteration"].max()]):
    select_emod_results = df_emod_results[df_emod_results["iteration"] <= iter_thresh]
    samples = select_emod_results[calib_vars].to_numpy()

    numpy_range = lambda x: (x.max() - x.min())
    cnrm = lambda x: (x - df_emod_results['cal_val'].min())/numpy_range(df_emod_results['cal_val'])


    # corner figure
    label_dict = dict(
        R0 = 'Reproductive\nNumber',
        R0_variance = 'Spreader\nHeterogeneity',
        indiv_variance_acq = 'Catcher\nHeterogeneity',
        correlation_acq_trans = 'Spreader-Catcher\nCorrelation'
    )
    fig = corner.corner(samples, labels=[label_dict[k] for k in calib_vars])

    # Extract the axes
    axes = np.array(fig.axes).reshape((ndim, ndim))


    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.scatter(select_emod_results[calib_vars[xi]], select_emod_results[calib_vars[yi]], 
                s = 3, c=cnrm(select_emod_results['cal_val']))

    plt.savefig(paths.figures / "calib_{}.png".format(ii), transparent=1)