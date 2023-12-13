#********************************************************************************
#
#********************************************************************************

import json

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



# ***** Setup *****
param_dict = dict()

param_dict['EXP_NAME']     = 'Demog-Workshop-Calib-04'
param_dict['NUM_SIMS']     =   1440
param_dict['NUM_ITER']     =     10
param_dict['EXP_VARIABLE'] = dict()
param_dict['EXP_CONSTANT'] = dict()
param_dict['EXP_OPTIMIZE'] = dict()


# Random number consistency
np.random.seed(4)

# Convenience naming
NSIMS = param_dict['NUM_SIMS']


# ***** Specify sim-variable parameters *****

param_dict['EXP_VARIABLE']['run_number']          =  list(range(NSIMS))


# ***** Parameters to auto-adjust *****

# Log of age-independent multiplier for mortality rates

param_dict['EXP_OPTIMIZE']['R0'] = [1.1, 1.9]
param_dict['EXP_OPTIMIZE']['R0_variance'] = [0, 1.5]
param_dict['EXP_OPTIMIZE']['indiv_variance_acq'] = [0, 1.5]
param_dict['EXP_OPTIMIZE']['correlation_acq_trans'] = [0, 1]


# ***** Constants for this experiment *****

# Number of days for simulation
param_dict['EXP_CONSTANT']['num_tsteps']   =  1000.0

# Calibrate to peak
param_dict['EXP_CONSTANT']['calibrate_peak']   =  1

# ***** Write parameter dictionary *****

with open('param_dict.json','w') as fid01:
  json.dump(param_dict,fid01)



#*******************************************************************************
