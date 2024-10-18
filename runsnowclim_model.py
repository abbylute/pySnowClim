import numpy as np
import pickle
from datetime import datetime, timedelta

"""
This script runs the SnowClim Model using an example of 4-hourly forcing data for the water year 2002
for an area in Central Idaho. It:
 - Sets default parameter values
 - Imports example forcing data
 - Runs the SnowClim model
 - Plots some results
 
The script can be adapted for different parameters, time periods, or datasets.
"""

# Check if the script is being run from the correct directory
import os
if not os.path.basename(os.getcwd()) == 'SnowClim-Model':
    raise RuntimeError('Please navigate to the SnowClim-Model directory.')

# Add the 'code' directory to the Python path
import sys
sys.path.append('code/')

# --- specify parameters ---
def create_parameter_file(cal, hours_in_ts, stability=None, windHt=None, tempHt=None,
                          snowoff_month=None, snowoff_day=None, albedo_option=None,
                          max_albedo=None, z_0=None, z_h=None, lw_max=None,
                          Tstart=None, Tadd=None, maxtax=None, E0=None, E0_app=None,
                          E0_stable=None, Ts_add=None, smooth_hr=None, ground_albedo=None,
                          snow_emis=None, snow_dens_default=None, G=None, parameterfilename=None):
    """
    Creates and saves a dictionary of snow model parameters.
    Default values are applied where parameters are not specified.
    """
    
    S = {
        'cal': cal,
        'hours_in_ts': hours_in_ts,
        'stability': 1 if stability is None else stability,
        'windHt': 10 if windHt is None else windHt,
        'tempHt': 2 if tempHt is None else tempHt,
        'snowoff_month': 9 if snowoff_month is None else snowoff_month,
        'snowoff_day': 1 if snowoff_day is None else snowoff_day,
        'albedo_option': 2 if albedo_option is None else albedo_option,
        'max_albedo': 0.85 if max_albedo is None else max_albedo,
        'z_0': 0.00001 if z_0 is None else z_0,
        'z_h': z_0 / 10 if z_h is None else z_h,
        'lw_max': 0.1 if lw_max is None else lw_max,
        'Tstart': 0 if Tstart is None else Tstart,
        'Tadd': -10000 if Tadd is None else Tadd,
        'maxtax': 0.9 if maxtax is None else maxtax,
        'E0_value': 1 if E0 is None else E0,
        'E0_app': 1 if E0_app is None else E0_app,
        'E0_stable': 2 if E0_stable is None else E0_stable,
        'Ts_add': 2 if Ts_add is None else Ts_add,
        'smooth_hr': 12 if smooth_hr is None else smooth_hr,
        'ground_albedo': 0.25 if ground_albedo is None else ground_albedo,
        'snow_emis': 0.98 if snow_emis is None else snow_emis,
        'snow_dens_default': 250 if snow_dens_default is None else snow_dens_default,
        'G': 173 / 86400 if G is None else G
    }
    
    # Save parameters to file
    with open(parameterfilename, 'wb') as f:
        pickle.dump(S, f)

# Time period for the model run (2001-10-01 to 2002-09-30)
hours_in_ts = 4
cal = np.array([[d.year, d.month, d.day, d.hour, d.minute, d.second]
                for d in (datetime(2001, 10, 1) + timedelta(hours=i * hours_in_ts)
                          for i in range(int((datetime(2002, 9, 30) - datetime(2001, 10, 1)).total_seconds() / 3600 / hours_in_ts)))])

# Other parameters (use default values for the rest)
parameterfilename = 'parameters.pkl'
create_parameter_file(cal, hours_in_ts, parameterfilename=parameterfilename)

# --- specify forcing data ---
data_dir = 'example_forcings/'

# Load example forcing data
latlonelev = np.load(f'{data_dir}lat_lon_elev.npy', allow_pickle=True).item()
lat = latlonelev['lat']
lon = latlonelev['lon']

# Load meteorological data (forcing inputs)
lrad = np.load(f'{data_dir}lrad.npy')
solar = np.load(f'{data_dir}solar.npy')
tavg = np.load(f'{data_dir}tavg.npy')
ppt = np.load(f'{data_dir}ppt.npy')
vs = np.load(f'{data_dir}vs.npy')
psfc = np.load(f'{data_dir}psfc.npy')
huss = np.load(f'{data_dir}huss.npy')
relhum = np.load(f'{data_dir}relhum.npy')
tdmean = np.load(f'{data_dir}tdmean.npy')

# --- run snow model ---
# snowclim_model is assumed to be implemented elsewhere
# Example model call:
# snowclim_model(lat, lrad, tavg, ppt, solar, tdmean, vs, relhum, psfc, huss, parameterfilename)

# --- run snow model ---
# Assuming the snowclim_model function is implemented and takes the required parameters.

(SnowMelt, SnowWaterEq, SFE, SnowDepth, SnowDensity, 
 Sublimation, Condensation, SnowTemp, MeltEnergy, Energy, Albedo, 
 SnowYN, RaininSnow, Runoff, RefrozenWater, PackWater, LW_down, LW_up, 
 SW_down, SW_up, Q_latent, Q_sensible, Q_precip, PackCC, CCenergy, 
 CCsnowfall) = snowclim_model(lat, lrad, tavg, ppt, solar, tdmean, 
                              vs, relhum, psfc, huss, parameterfilename)

