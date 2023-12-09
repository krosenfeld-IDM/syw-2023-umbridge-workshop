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

param_dict['EXP_NAME']     = 'Workshop_1d_00'
param_dict['NUM_SIMS']     =  6*400
param_dict['EXP_VARIABLE'] = dict()
param_dict['EXP_CONSTANT'] = dict()

# Random number consistency
np.random.seed(4)

# Convenience naming
NSIMS = param_dict['NUM_SIMS']


# ***** Specify sim-variable parameters *****

p_levels = [[  0.0],
            [  0.0],
            [  0.0]]
rand_lev = np.random.randint(0,len(p_levels[0]),size=NSIMS).tolist()

R0_ranges = [[1.3, 1.8], [1.4, 2.2], [1.5, 2], [1.7, 1.8], [1.5, 2]]
names = ['Flu', 'Mumps', 'Smallpox', 'Diptheria', 'Rubella']

param_dict['EXP_VARIABLE']['run_number']             =     list(range(NSIMS))

# R0 values for tranmssion
num_sims_per_R0 = int(param_dict['NUM_SIMS']/len(R0_ranges))
R0s = np.array([np.random.randn(num_sims_per_R0)*((R0_range[1]-R0_range[0])/3) + np.mean(R0_range) for R0_range in R0_ranges]).flatten()
param_dict['EXP_VARIABLE']['R0']                     =     R0s.tolist()

# Name of disease associated with R0
param_dict['EXP_VARIABLE']['R0_name'] = np.array([[name]*num_sims_per_R0 for name in names]).flatten().tolist()

# R0 variance; (log-normal distribution)
param_dict['EXP_VARIABLE']['R0_variance']            =     [p_levels[0][val] for val in rand_lev]

# Individual level acquisition variance; (mean = 1.0; log-normal distribution)
param_dict['EXP_VARIABLE']['indiv_variance_acq']     =     [p_levels[1][val] for val in rand_lev]

# Acquision-transmission correlation;
param_dict['EXP_VARIABLE']['correlation_acq_trans']  =     [p_levels[2][val] for val in rand_lev]


# ***** Constants for this experiment *****

# Number of days for simulation;

param_dict['EXP_CONSTANT']['num_tsteps']           =  1000.0

param_dict['EXP_CONSTANT']['return_timeseries']    = True

# ***** Write parameter dictionary *****

with open('param_dict.json','w') as fid01:
  json.dump(param_dict,fid01)



#*******************************************************************************
