#********************************************************************************
#
#********************************************************************************

import os, sys, json

import global_data as gdata

import numpy as np

#********************************************************************************

def application(output_path):

  SIM_IDX       = gdata.sim_index


  # Prep output dictionary
  key_str    = '{:05d}'.format(SIM_IDX)
  parsed_dat = {key_str: dict()}


  # Calculate total attack rate and store in a json dict
  with open(os.path.join(output_path,'InsetChart.json')) as fid01:
    inset_chart = json.load(fid01)

  new_inf = np.array(inset_chart['Channels']['New Infections']['Data'])
  pop_vec = np.array(inset_chart['Channels']['Statistical Population']['Data'])

  tot_pop = pop_vec[-1]
  max_inf = np.argmax(new_inf)
  tot_inf = np.sum(new_inf)
  epi_inf = np.sum(new_inf[:max_inf])

  parsed_dat[key_str]['atk_frac']  = int(tot_inf)/tot_pop
  parsed_dat[key_str]['herd_frac'] = int(epi_inf)/tot_pop

  # assessment metric for calibration (bigger is better --> 0)
  # https://github.com/InstituteforDiseaseModeling/EMOD-Generic-Scripts/blob/main/model_demographics01/Assets/python/dtk_post_process.py
  parsed_dat[key_str]['cal_val'] = float(-np.abs(int(tot_inf)/tot_pop - 0.4) / (2*0.05**2))

  if gdata.var_params.get('calibrate_peak', False):
    parsed_dat[key_str]['cal_val'] += float(-np.abs(np.argmax(np.cumsum(new_inf)) - 175) / (2*10**2))

  if gdata.var_params.get('return_timeseries', False):
    # Monthly timeseries
    parsed_dat[key_str]['timeseries']   = new_inf.tolist()

  # Write output dictionary
  with open('parsed_out.json','w') as fid01:
    json.dump(parsed_dat, fid01)


  return None

#*******************************************************************************
