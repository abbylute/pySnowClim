"""
This module contains functions to calculate snow albedo using different snow models: 
Essery et al. (2013), the Utah Snow Model (Tarboton), and the VIC model. These models 
compute the albedo based on snow properties such as snow depth, snow water equivalent 
(SWE), snow temperature, and snow age, along with environmental conditions like latitude, 
ground albedo, and time of year.

Main Function:
--------------
- calc_albedo: The central function to compute snow albedo based on the selected model option.

Albedo Calculation Models:
---------------------------
1. Essery et al. (2013) Option 1:
   - Calculates snow albedo based on fresh snow, cold snow, and melting snow conditions 
     using equations from Essery et al. (2013), with parameter values from Douville et al. (1995).
   
2. Utah Snow Model (Tarboton):
   - Albedo is calculated using equations adapted from the Utah Energy Balance (UEB) model.
   - Albedo is a function of snow age, temperature, and new snow depth, taking into account 
     latitude, month, and day.

3. VIC Model:
   - The VIC (Variable Infiltration Capacity) model calculates albedo for new snow, cold-aged snow, 
     and melting snow conditions, using snow depth and snow age. The model was adapted from the 
     VIC snow utility.
"""

import numpy as np

def calc_albedo(albedo_option, ground_albedo, max_albedo, last_albedo, new_snow_depth, last_snow_depth, new_swe, last_swe, last_snow_temp, water_dens, lat, month, day, snow_age, last_pack_cc, sec_in_ts):
    """
    Calculate the snow albedo based on the selected model option and various snow and environmental parameters.

    Parameters:
    -----------
    albedo_option : int
        The albedo calculation method to use. Options include:
        1 = Essery et al. (2013) option 1
        2 = Utah Snow Model (Tarboton)
        3 = VIC model
    ground_albedo : float
        The albedo of the ground surface.
    max_albedo : float
        Maximum snow albedo.
    last_albedo : ndarray
        Previous timestep snow albedo values.
    new_snow_depth : ndarray
        Depth of newly fallen snow.
    last_snow_depth : ndarray
        Depth of snow from the previous timestep.
    new_swe : ndarray
        Snow Water Equivalent (SWE) of new snow.
    last_swe : ndarray
        SWE from the previous timestep.
    last_snow_temp : ndarray
        Snow temperature from the previous timestep.
    water_dens : float
        Density of water (kg/m^3).
    lat : ndarray
        Latitude of the snow surface points.
    month : int
        Month of the current timestep (1-12).
    day : int
        Day of the current timestep (1-31).
    snow_age : ndarray
        Age of the snowpack.
    last_pack_cc : ndarray
        Cold content of the snowpack from the previous timestep.
    sec_in_ts : int
        Number of seconds in the current timestep.

    Returns:
    --------
    albedo : ndarray
        Calculated snow albedo for each grid point.
    snow_age : ndarray
        Updated snow age for each grid point.
    """
    
    if albedo_option == 1:  # Essery et al. (2013) option 1
        albedo = calc_albedo_essery_opt1(last_albedo, last_snow_temp, new_swe, last_swe, max_albedo, ground_albedo, water_dens, sec_in_ts)
        
    elif albedo_option == 2:  # Utah Snow Model (Tarboton)
        albedo, snow_age = calc_albedo_tarboton(last_snow_temp + 273.15, snow_age, new_snow_depth, lat, month, day, last_swe, ground_albedo, max_albedo, sec_in_ts)
        
    elif albedo_option == 3:  # VIC model
        albedo, snow_age = calc_albedo_vic(new_snow_depth, snow_age, ground_albedo, last_snow_depth, last_albedo, max_albedo, sec_in_ts, last_pack_cc)

    # Adjust albedo if total snow depth is low (< 0.1 m)
    z = last_snow_depth
    f = z < 0.1
    r = (1 - z[f] / 0.1) * np.exp(-z[f] / 0.2)
    r = np.minimum(r, 1)
    albedo[f] = r * ground_albedo + (1 - r) * albedo[f]

    # Ensure albedo is real and within range
    albedo = np.real(albedo)
    albedo = np.clip(albedo, 0, max_albedo)

    return albedo, snow_age


def calc_albedo_essery_opt1(last_albedo, last_snow_temp, new_swe, last_swe, max_albedo, ground_albedo, water_dens, sec_in_ts):
    """
    Calculate snow albedo based on Essery et al. (2013) option 1.

    Parameters are similar to those in `calc_albedo`.

    Returns:
    --------
    albedo : ndarray
        Calculated snow albedo.
    """
    min_albedo = 0.5  # minimum snow albedo
    So = 10  # kg/m² ('critical SWE')
    Ta = 1e7  # s
    Tm = 3.6e5  # s

    dt = sec_in_ts
    Sf = new_swe * water_dens / dt  # snow fall rate (kg m-2 s-1)

    albedo = np.zeros_like(last_albedo)

    # No snow on the ground
    b = last_swe == 0
    albedo[b] = ground_albedo

    # Fresh snow
    b = new_swe > 0
    albedo[b] = last_albedo[b] + (max_albedo - last_albedo[b]) * ((Sf[b] * dt) / So)

    # Cold snow
    b = (last_swe > 0) & (new_swe == 0) & (last_snow_temp < -0.5)
    albedo[b] = last_albedo[b] - dt / Ta

    # Melting snow
    b = (last_swe > 0) & (new_swe == 0) & (last_snow_temp >= -0.5)
    albedo[b] = (last_albedo[b] - min_albedo) * np.exp(-dt / Tm) + min_albedo

    albedo = np.clip(albedo, min_albedo, max_albedo)

    return albedo


def calc_albedo_vic(new_snow_depth, snow_age, ground_albedo, last_snow_depth, last_albedo, max_albedo, sec_in_ts, last_pack_cc):
    """
    Calculate snow albedo based on the VIC model.

    Parameters are similar to those in `calc_albedo`.

    Returns:
    --------
    albedo : ndarray
        Calculated snow albedo.
    snow_age : ndarray
        Updated snow age.
    """
    SNOW_NEW_SNOW_ALB = max_albedo
    SNOW_ALB_ACCUM_A = 0.94
    SNOW_ALB_ACCUM_B = 0.58
    SNOW_ALB_THAW_A = 0.82
    SNOW_ALB_THAW_B = 0.46
    sec_per_day = 24 * 60 * 60

    albedo = np.copy(last_albedo)

    # New snow case
    b = (new_snow_depth > 0.01) & (last_pack_cc < 0)
    snow_age[b] = 0
    albedo[b] = SNOW_NEW_SNOW_ALB

    # Aged snow case
    b = ~b & (last_snow_depth > 0)
    snow_age[b] += sec_in_ts

    # Cold snow
    c = b & (last_pack_cc < 0)
    albedo[c] = SNOW_NEW_SNOW_ALB * SNOW_ALB_ACCUM_A ** ((snow_age[c] / sec_per_day) ** SNOW_ALB_ACCUM_B)

    # Melting snow
    c = b & (last_pack_cc == 0)
    albedo[c] = SNOW_NEW_SNOW_ALB * SNOW_ALB_THAW_A ** ((snow_age[c] / sec_per_day) ** SNOW_ALB_THAW_B)

    # No snow case
    b = last_snow_depth == 0
    snow_age[b] = 0
    albedo[b] = ground_albedo

    return albedo, snow_age


def calc_albedo_tarboton(last_snow_temp, snow_age, new_snow_depth, lat, month, day, last_swe, ground_albedo, max_albedo, sec_in_ts):
    """
    Calculate snow albedo based on the Utah Snow Model (Tarboton).

    Parameters are similar to those in `calc_albedo`.

    Returns:
    --------
    albedo : ndarray
        Calculated snow albedo.
    snow_age : ndarray
        Updated snow age.
    """
    Cv = 0.2
    Cir = 0.5

    albedo_iro = 0.65
    albedo_vo = 2 * max_albedo - albedo_iro
    if albedo_vo > 1:
        d = albedo_vo - 1
        albedo_iro += d
        albedo_vo = 1

    albedo = np.zeros_like(last_swe)

    # New snow depth > 0.01m
    b = new_snow_depth > 0.01
    albedo[b] = albedo_vo / 2 + albedo_iro / 2
    snow_age[b] = 0

    # No snow case
    b = last_swe == 0
    albedo[b] = ground_albedo
    snow_age[b] = 0

    return albedo, snow_age
