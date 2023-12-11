#********************************************************************************
#
#********************************************************************************

import json
import pyDOE2 as pydoe

import numpy as np

#*******************************************************************************

# This script makes a json dictionary that is used by the pre-processing script
# in EMOD. Variable names defined here will be available to use in creating
# the input files. Please don't change the variable name for 'EXP_NAME' or
# for 'NUM_SIMS' because those are also used in scripts outside of EMOD.

# The pre-process script will open the json dict created by this method. For
# everything in the 'EXP_VARIABLE' key, that script will assume a list and
# get a value from that list based on the sim index. For everything in the 
# 'EXP_CONSTANT' key, it will assume a single value and copy that value.

n_rand = 400
R0_variance = [0.0, 0.3, 0.5, 0.7]
indiv_variance_acq = [0.0, 0.3, 0.5, 0.7]
correlation_acq_trans = [0.0, 0.3, 0.5, 0.7]
levels = [len(R0_variance), len(indiv_variance_acq), len(correlation_acq_trans)]
design_mat = np.tile(pydoe.fullfact(levels).astype(int), (n_rand,1))

# ***** Setup *****
param_dict = dict()

param_dict['EXP_NAME']     = 'Workshop_correlation01'
param_dict['NUM_SIMS']     =  len(design_mat)
param_dict['EXP_VARIABLE'] = dict()
param_dict['EXP_CONSTANT'] = dict()

# Random number consistency
np.random.seed(4)

# Convenience naming
NSIMS = param_dict['NUM_SIMS']


# ***** Specify sim-variable parameters *****

param_dict['EXP_VARIABLE']['run_number']             =     list(range(NSIMS))

# R0 values for tranmssion
param_dict['EXP_VARIABLE']['R0']                     =     np.random.uniform(low= 0.50,high= 1.75, size=NSIMS).tolist()

# R0 variance; (log-normal distribution)
param_dict['EXP_VARIABLE']['R0_variance']            =     [R0_variance[entry[0]] for entry in design_mat]

# Individual level acquisition variance; (mean = 1.0; log-normal distribution)
param_dict['EXP_VARIABLE']['indiv_variance_acq']     =     [indiv_variance_acq[entry[1]] for entry in design_mat]

# Acquision-transmission correlation;
param_dict['EXP_VARIABLE']['correlation_acq_trans']  =     [correlation_acq_trans[entry[2]] for entry in design_mat]

# ***** Constants for this experiment *****

# Number of days for simulation;
param_dict['EXP_CONSTANT']['num_tsteps']           =  1000.0

param_dict['EXP_CONSTANT']['return_timeseries']    = False

# ***** Write parameter dictionary *****

with open('param_dict.json','w') as fid01:
  json.dump(param_dict,fid01)



#*******************************************************************************
