"""
Look at the role of heterogeneity in transmission and acquisition
"""

import matplotlib.pyplot as plt
import numpy as np
import paths
import methods
from sklearn.linear_model import LogisticRegression

# load simulation results
emod_results = methods.DataBrick(paths.emod / 'model_covariance01' / 'experiment_workshop_correlation_00' ).to_df()

plt.figure()
ii = 0
for n0, g0 in emod_results.groupby('R0_variance'):
    for n1, g1 in g0.groupby('indiv_variance_acq'):
        plt.plot(g1['R0'], g1['atk_frac'], 'o', label=f'{n0}, {n1}', color='C{:d}'.format(ii))

        # logistic regression
        R0 = np.array(g1['R0'])[:,None]        
        outbreak = np.array(1*(g1['atk_frac'] > 0.001)).ravel()
        # Create a logistic regression model
        model = LogisticRegression()

        # Train the model
        model.fit(R0, outbreak.ravel())

        # Make predictions
        ax = plt.gca()
        x = np.linspace(*ax.get_xlim())[:, None]
        y_pred = model.predict_proba(x)
        ax.plot(x, y_pred[:,1], '-', color='C{:d}'.format(ii))
        ii += 1

plt.legend()
plt.savefig(paths.figures / 'tmp.png')

# #######################################################
# fig, axes = plt.subplots(1,3,figsize=(12,5))
# ii = 0
# for n0, g0 in emod_results.groupby('R0_variance'):
#     for n1, g1 in g0.groupby('indiv_variance_acq'):
#         ax = axes[ii]
#         ax.plot(g1['R0'], g1['atk_frac'], 'o', label=f'{n0}, {n1}')

#         R0 = np.array(g1['R0'])[:,None]        
#         outbreak = np.array(1*(g1['atk_frac'] > 0.01))[:, None]

#         # sort R0 and outbreak by R0
#         # ii = np.argsort(R0[:,0])
#         # R0 = R0[ii]
#         # outbreak = outbreak[ii]

#         # Create a logistic regression model
#         model = LogisticRegression()

#         # Train the model
#         model.fit(R0, outbreak.ravel())

#         # Make predictions
#         x = np.linspace(*ax.get_xlim())[:, None]
#         y_pred = model.predict_proba(x)
#         ax.plot(R0, outbreak, 'o')
#         ax.plot(x, y_pred[:,1], '-')
#         ax.legend()
#         ii += 1
# plt.savefig(paths.figures / 'tmp.png')