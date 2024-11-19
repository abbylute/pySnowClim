import numpy as np
import pickle

import constants as const

from calcPhase import calc_phase
from calcFreshSnowDensity import calc_fresh_snow_density
from calcSnowDensityAfterSnow import calc_snow_density_after_snow
from calcSnowDensity import calc_snow_density
from updatePackWater import update_pack_water
from calcAlbedo import calc_albedo
from calcTurbulentFluxes import calc_turbulent_fluxes
from calcLongwave import calc_longwave
from calcEnergyToCC import calc_energy_to_cc
from calcEnergyToRefreezing import calc_energy_to_refreezing
from calcEnergyToMelt import calc_energy_to_melt
from calcSublimation import calc_sublimation

from SnowModelVariables import SnowModelVariables
from PrecipitationProperties import PrecipitationProperties
from SnowpackVariables import Snowpack

def _prepare_outputs(model_vars, precip):
    """
    Prepare outputs by scaling model variables with appropriate constants.

    Args:
    -----
    - model_vars: object
        An object containing model variables such as SnowWaterEq, SnowDepth, etc.
    - precip: Class
        Propreties of precipitation, including swe.

    Returns:
    --------
    Updated model_vars with scaled values for output.
    """
    # Scale model variables by WATERDENS or other units where applicable
    model_vars.SnowWaterEq *= const.WATERDENS
    model_vars.SnowfallWaterEq = precip.sfe * const.WATERDENS
    model_vars.SnowMelt *= const.WATERDENS
    model_vars.Sublimation *= const.WATERDENS
    model_vars.Condensation *= const.WATERDENS
    model_vars.SnowDepth *= 1000  # Convert to mm
    model_vars.Runoff *= const.WATERDENS
    model_vars.RaininSnow *= const.WATERDENS
    model_vars.RefrozenWater *= const.WATERDENS
    model_vars.PackWater *= const.WATERDENS

    return model_vars


def _perform_precipitation_operations(forcings_data, parameters):
    """
    Perform precipitation phase calculations and adjust snow and rain values.

    Args:
    -----
    - forcings_data: dict
        A dictionary containing forcing data, including 'tavg' (temperature) and 'relhum' (relative humidity).
    - parameters: dict
        Model parameters, including 'hours_in_ts' (hours per time step).

    Returns:
    --------
    - rainfall: np.ndarray
        Rainfall amount after phase calculation.
    - SnowfallWaterEq: np.ndarray
        Snowfall equivalent after phase calculation.
    - newsnowdensity: np.ndarray
        Fresh snow density based on the average temperature.
    """
    # Calculate phase (snow or rain fraction)
    passnow = calc_phase(forcings_data['tavg'], forcings_data['relhum'])

    # Separate rain and snow components of precipitation
    rainfall = forcings_data['ppt'] * (1 - passnow)
    SnowfallWaterEq = forcings_data['ppt'] * passnow

    # Threshold for snowfall equivalent and adjust rain accordingly
    threshold = 0.0001 * parameters['hours_in_ts']
    SnowfallWaterEq[SnowfallWaterEq < threshold] = 0
    rainfall[SnowfallWaterEq < threshold] = forcings_data['ppt'][SnowfallWaterEq < threshold]

    # Calculate fresh snow density based on temperature
    newsnowdensity = calc_fresh_snow_density(forcings_data['tavg'])
    
    precip = PrecipitationProperties(rainfall, SnowfallWaterEq, newsnowdensity, 
                                     const.WATERDENS)
    return precip


def _process_forcings_and_energy(index, forcings_data, parameters, 
                                 snow_model_instances):
    """
    Processes input forcings and smooths energy calculations.
    
    Parameters:
    - index: int, index of the current timestep.
    - forcings_data: dict, containing forcings data for the model.
    - parameters: dict, containing model parameters.
    - snow_model_instances: list, previous instances of SnowModelVariables containing energy data.

    Returns:
    - input_forcings: dict, forcings data for the current timestep.
    - snow_vars: SnowModelVariables, initialized snow model variables for current timestep.
    - smoothed_energy: ndarray, energy values smoothed over specified hours.
    """

    # Extract current timestep forcings
    input_forcings = {key: value[index, :] for key, value in forcings_data['forcings'].items()}

    # Initialize snow model variables for this timestep
    size_lat = forcings_data['coords']['lat'].size
    snow_vars = SnowModelVariables(size_lat)

    smoothed_energy = None
    if index > parameters['smooth_time_steps']:        
        smoothed_energy = np.full((parameters['smooth_time_steps'], size_lat), np.nan)
        for l in range(parameters['smooth_time_steps']):
            smoothed_energy[l, :] = snow_model_instances[index -l -1].Energy
    else:
        if index > 0:
            smoothed_energy = np.full((index+1, size_lat), np.nan)
            for l in range(index):
                smoothed_energy[l, :] = snow_model_instances[l].Energy

    return input_forcings, snow_vars, smoothed_energy


def _calculate_snow_temp_and_cold_content(newswe, input_forcings, snowpack):
    """
    Calculates snow temperature and cold content and updates pack conditions.
    
    Parameters:
    - newswe: ndarray, snow water equivalent for the current timestep.
    - input_forcings: dict, containing current timestep forcings such as temperature.
    - lastpackcc: ndarray, previous timestep's cold content.
    - lastswe: ndarray, previous timestep's snow water equivalent.
    - lastpacktemp: ndarray, temperature of the last snow pack.

    Returns:
    - snowfallcc: ndarray, with the snowlfall cold content.
    - lastpacktemp: ndarray, updated temperature for last pack.
    """
    has_new_snow = newswe > 0

    # Update new snow temperature where there is new snow
    newsnowtemp = np.where(has_new_snow, np.minimum(0, input_forcings['tdmean']), 0)

    # Calculate the cold content of the snowfall
    snowfallcc = const.WATERDENS * const.CI * newswe * newsnowtemp
    snowpack.lastpackcc += snowfallcc.copy()

    # Update last pack temperature where there is snowfall
    if np.any(has_new_snow):
        snowpack.lastpacktemp[has_new_snow] = snowpack.lastpackcc[has_new_snow] / \
            (const.WATERDENS * const.CI * (snowpack.lastswe[has_new_snow] + newswe[has_new_snow]))

    return snowfallcc


def _calculate_energy_fluxes(exist_snow, parameters, input_forcings, lastsnowtemp, newrain, 
                            lastalbedo, sec_in_ts):
    """
    Calculate energy fluxes at each timestep, including turbulent, rain, solar, longwave, and ground heat fluxes.

    Parameters:
    - exist_snow: Boolean array, indicating locations with existing snow.
    - parameters: dict, contains model parameters like 'G' and 'snow_emis'.
    - input_forcings: dict, contains the forcing data for this timestep.
    - lastsnowtemp: ndarray, surface temperature of the last snowpack.
    - newrain: ndarray, rainfall over snowpack.
    - lastalbedo: ndarray, albedo values from the previous timestep.
    - sec_in_ts: float, number of seconds in each timestep.

    Returns:
    - lastenergy: ndarray, net downward energy flux into the snow surface.
    """
    var_list = ['Q_sensible', 'E', 'Q_latent', 'Q_precip', 'SW_up', 'SW_down', 'LW_down', 
                'LW_up']
    energy_var = {name: np.zeros(len(exist_snow), dtype=np.float32) for name in var_list}

    # --- Calculate turbulent heat fluxes (kJ/m2/timestep) ---    
    has_snow_and_wind = np.logical_and(exist_snow, (input_forcings['vs'] > 0))
    H, E, EV = calc_turbulent_fluxes(
        parameters, input_forcings['vs'][has_snow_and_wind], lastsnowtemp[has_snow_and_wind],
        input_forcings['tavg'][has_snow_and_wind], input_forcings['psfc'][has_snow_and_wind],
        input_forcings['huss'][has_snow_and_wind], sec_in_ts
    )
    energy_var['Q_sensible'][has_snow_and_wind] = H
    energy_var['E'][has_snow_and_wind] = E
    energy_var['Q_latent'][has_snow_and_wind] = EV

    # --- Rain heat flux into snowpack (kJ/m2/timestep) ---
    energy_var['Q_precip'][exist_snow] = const.CW * const.WATERDENS * np.maximum(0, input_forcings['tdmean'][exist_snow]) * newrain[exist_snow]

    # --- Net downward solar flux at surface (kJ/m2/timestep) ---
    energy_var['SW_up'][exist_snow] = input_forcings['solar'][exist_snow] * lastalbedo[exist_snow]
    energy_var['SW_down'][exist_snow] = input_forcings['solar'][exist_snow]

    # --- Longwave flux up from snow surface (kJ/m2/timestep) ---
    energy_var['LW_down'][exist_snow] = input_forcings['lrad'][exist_snow]
    energy_var['LW_up'][exist_snow] = calc_longwave(
        parameters['snow_emis'], lastsnowtemp[exist_snow], 
        input_forcings['lrad'][exist_snow], sec_in_ts)

    # --- Ground heat flux (kJ/m2/timestep) ---
    energy_var['Gf'] = np.where(exist_snow, parameters['G'] * sec_in_ts, 0)

    return energy_var


def _apply_cold_content_tax(lastpackcc, parameters, previous_energy, lastenergy):
    """
    Apply cold content tax to the energy flux based on temperature parameters.
    
    Parameters:
    - lastpackcc (np.ndarray): The current cold content in the snowpack.
    - parameters (dict): Dictionary containing temperature and tax parameters, 
                         including 'Tstart', 'Tadd', and 'maxtax'.
    - previous_energy (np.ndarray): The smoothed energy values.

    Returns:
    - np.ndarray: Updated energy values with the cold content tax applied.
    """
    # Calculate the tax based on the cold content
    tax = (lastpackcc - parameters['Tstart']) / parameters['Tadd'] * parameters['maxtax']
    # Limit tax to be within 0 and maxtax
    tax = np.clip(tax, 0, parameters['maxtax'])

    if parameters['smooth_time_steps'] > 1 and previous_energy is not None:
        previous_energy[-1,:] = lastenergy
        smoothed_energy = np.nanmean(previous_energy, axis=0)
    else:
        smoothed_energy = lastenergy
        
    # Apply tax where energy is negative
    negative_energy = smoothed_energy < 0
    if np.any(negative_energy):
        smoothed_energy[negative_energy] *= (1 - tax[negative_energy])

    return smoothed_energy


def _update_snowpack_state(
    input_forcings, parameters, snow_vars,  
    precip, snowpack, sec_in_ts,  
    coords, time_value, runoff
):
    """
    Updates the snowpack state variables based on surface temperature, snowfall,
    pack density, water content, and albedo.

    Args:
        input_forcings (dict): Dictionary with temperature and other input forcings.
        parameters (dict): Model parameters for snowpack calculations.
        snow_vars (object): Snow model variables object for storing outputs.
        precip (class): Precipitation properties.
        lastpacktemp (np.ndarray): Previous pack temperature.
        sec_in_ts (float): Number of seconds in a timestep.
        coords (dict): Coordinates information.
        time_value (tuple): Current time step and other time-related info.
        runoff (np.ndarray): Runoff.

    Returns:
        updated_runoff (np.ndarray): Updated runoff.
    """
    
    # Update snowpack after new snowfall
    snowpack.packsnowdensity[snow_vars.ExistSnow] = calc_snow_density_after_snow(
        snowpack.lastswe[snow_vars.ExistSnow], precip.sfe[snow_vars.ExistSnow], 
        snowpack.packsnowdensity[snow_vars.ExistSnow], precip.snowdens[snow_vars.ExistSnow]
    )
    
    snowpack.lastswe += precip.sfe.copy()
    snowpack.lastsnowdepth += precip.snowdepth.copy()

    # Calculate pack density after compaction
    snowpack.packsnowdensity[snow_vars.ExistSnow] = calc_snow_density(
        snowpack.lastswe[snow_vars.ExistSnow], snowpack.lastpacktemp[snow_vars.ExistSnow], 
        snowpack.packsnowdensity[snow_vars.ExistSnow], sec_in_ts
    )
    snowpack.lastsnowdepth[snow_vars.ExistSnow] = snowpack.lastswe[snow_vars.ExistSnow] * const.WATERDENS / snowpack.packsnowdensity[snow_vars.ExistSnow]

    # Update snowpack liquid water content
    previouspackwater = snowpack.lastpackwater.copy()
    snowpack.lastpackwater[snow_vars.ExistSnow] += precip.rain[snow_vars.ExistSnow]
    updated_runoff, lastpackwater = update_pack_water(
        snow_vars.ExistSnow, snowpack.lastpackwater, snowpack.lastsnowdepth, 
        parameters['lw_max'], runoff, sec_in_ts
    )
    snowpack.lastpackwater = lastpackwater.copy()
    snowpack.rain_in_snow = np.where(snow_vars.ExistSnow, 
                                     np.maximum(snowpack.lastpackwater - previouspackwater, 0), 
                                     np.nan)

    lastalbedo, snowage = calc_albedo(
        parameters, snowpack.lastalbedo, precip.snowdepth, snowpack.lastsnowdepth, 
        precip.sfe, snowpack.lastswe,
        snow_vars.SnowTemp, coords['lat'].ravel(), time_value[1], time_value[2],
        snowpack.snowage, snowpack.lastpackcc, sec_in_ts
    )
    snowpack.snowage = snowage.copy()
    snowpack.lastalbedo = lastalbedo.copy()

    return updated_runoff


def _apply_snowpack_temperature_instability_correction(snowpack, input_forcings, 
                                                       sec_in_ts):
    """
    Apply a temperature instability correction to adjust lastpacktemp and lastpackcc 
    for snow with small SWE values.

    Args:
        lastswe (np.ndarray): Snow Water Equivalent at the current timestep.
        lastpacktemp (np.ndarray): Temperature of the snowpack.
        lastpackcc (np.ndarray): Cold content of the snowpack.
        input_forcings (dict): Input meteorological forcings, including 'tavg'.
        sec_in_ts (float): Number of seconds in the timestep.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Updated lastpacktemp and lastpackcc arrays.
    """
    # Define threshold based on time step
    thres = sec_in_ts / const.HR_2_SECS * 0.015  # 15mm for each hour in the time step

    # Identify indices where the instability correction should apply
    instability_indices = np.where((snowpack.lastswe < thres) & (snowpack.lastswe > 0) & (snowpack.lastpacktemp < input_forcings['tavg']))

    # Apply corrections
    snowpack.lastpacktemp[instability_indices] = np.minimum(0, input_forcings['tavg'][instability_indices])
    snowpack.lastpackcc[instability_indices] = const.WATERDENS * const.CI * snowpack.lastswe[instability_indices] * snowpack.lastpacktemp[instability_indices]



def _distribute_energy_to_snowpack(taxed_last_energy, snow_vars,  snowpack, sec_in_ts,
                                   input_forcings, parameters):
    """
    Distributes the available energy to the snowpack components: cold content, refreezing, and melting.

    Args:
        taxed_last_energy (np.ndarray): Taxed energy available for distribution.
        snow_vars (object): Snow variables object to update (e.g., CCenergy, SnowMelt, etc.).
        snowpack (object): Snowpack variables class (e.g., lastsnowdepth, packsnowdensity, etc.).
        sec_in_ts (float): Number of seconds in the timestep.
        input_forcings (dict): Input meteorological forcings (e.g., temperature, tavg).
        parameters (dict): Parameters for snowpack calculations (e.g., snow density, melt coefficients).

    Returns:
        Updated snow_vars object, and modified lastpackcc, taxed_last_energy, lastpacktemp, lastpackwater, and lastswe.
    """
    # 1. Energy goes to cold content
    snowpack.lastpackcc, taxed_last_energy, snow_vars.CCenergy = calc_energy_to_cc(
        snowpack.lastpackcc.copy(), taxed_last_energy.copy(), snow_vars.CCenergy.copy())

    # Temperature adjustment for snow with positive SWE
    b = snowpack.lastswe > 0
    if np.any(b):
        snowpack.lastpacktemp[b] = snowpack.lastpackcc[b] / (const.WATERDENS * const.CI * snowpack.lastswe[b])

    _apply_snowpack_temperature_instability_correction(snowpack, input_forcings, 
                                                       sec_in_ts)
    
    # 2. Energy goes to refreezing
    lastpackwater, lastswe, lastpackcc, packsnowdensity, snow_vars.RefrozenWater = calc_energy_to_refreezing(
        snowpack.lastpackwater.copy(), snowpack.lastswe.copy(), snowpack.lastpackcc.copy(), 
        snowpack.lastsnowdepth.copy(), snow_vars.RefrozenWater.copy(), snowpack.packsnowdensity.copy())
    snowpack.lastpackcc = lastpackcc.copy()
    
    # 3. Energy goes to melting
    snow_vars.SnowMelt, snow_vars.MeltEnergy, lastpackwater, lastswe, lastsnowdepth, lastenergy  = calc_energy_to_melt(
        lastswe, snowpack.lastsnowdepth.copy(), packsnowdensity, taxed_last_energy, 
        lastpackwater, snow_vars.SnowMelt.copy(), snow_vars.MeltEnergy.copy())
    snowpack.lastswe = lastswe.copy()
    snowpack.lastsnowdepth = lastsnowdepth.copy()
    snowpack.packsnowdensity = packsnowdensity.copy()

    # Update water in snowpack
    snow_vars.Runoff, lastpackwater = update_pack_water(
        snow_vars.ExistSnow, lastpackwater, lastsnowdepth, parameters['lw_max'], 
        snow_vars.Runoff, sec_in_ts)
    snow_vars.PackWater = lastpackwater.copy()
    snowpack.lastpackwater = lastpackwater.copy()

    return snow_vars


def _process_snowpack(input_forcings, parameters, snow_vars, 
                      precip, snowpack, sec_in_ts, 
                      coords, time_value, previous_energy):
    """
    Processes snowpack state, energy fluxes, and updates sublimation, condensation, 
    and snow variables after precipitation and energy adjustments.    
    """
    updated_runoff = _update_snowpack_state(
         input_forcings, parameters, snow_vars, precip, snowpack, sec_in_ts, coords, 
         time_value, snow_vars.Runoff)
             
    energy_var = _calculate_energy_fluxes(
        snow_vars.ExistSnow, parameters, input_forcings, snow_vars.SnowTemp, precip.rain, 
        snowpack.lastalbedo, sec_in_ts)
    
    # --- Downward net energy flux into snow surface (kJ/m2/timestep) ---
    lastenergy = energy_var['SW_down'] - energy_var['SW_up']  
    lastenergy += energy_var['LW_down'] - energy_var['LW_up'] 
    lastenergy += energy_var['Q_sensible'] + energy_var['Q_latent']  
    lastenergy += energy_var['Gf'] + energy_var['Q_precip']

    # copying values from the dict to the object
    for key, value in energy_var.items():
        if hasattr(snow_vars, key):
            setattr(snow_vars, key, value)


    snow_vars.Energy = lastenergy.copy()            

    taxed_last_energy = _apply_cold_content_tax(
        snowpack.lastpackcc, parameters, previous_energy, lastenergy)         

    snow_vars = _distribute_energy_to_snowpack(
         taxed_last_energy, snow_vars, snowpack, sec_in_ts, input_forcings, parameters)
    
    # --- Sublimation ---
    a = snowpack.lastsnowdepth > 0
    if np.any(a):
        snow_vars.Sublimation[a], snow_vars.Condensation[a], snowpack.lastswe[a], snowpack.lastsnowdepth[a], \
            snowpack.lastpackcc[a], snowpack.packsnowdensity[a], \
            snowpack.lastpackwater[a] = calc_sublimation(
                energy_var['E'][a], snowpack.lastswe[a], snowpack.lastsnowdepth[a], 
                snowpack.packsnowdensity[a],
                snow_vars.SnowTemp[a], snowpack.lastpackcc[a], parameters['snow_dens_default'],
                snow_vars.Sublimation[a], snow_vars.Condensation[a],
                snowpack.lastpackwater[a])

    # Update snow
    b = snowpack.lastswe > 0
    snowpack.lastpackwater[~b] = 0
    snowpack.lastalbedo[~b] = parameters['ground_albedo']
    snowpack.snowage[~b] = 0
    snowpack.lastpacktemp[~b] = 0
    snowpack.lastpackcc[~b] = 0
    snowpack.lastsnowdepth[~b] = 0
    snowpack.lastpacktemp[b] = snowpack.lastpackcc[b] / (const.WATERDENS * const.CI * snowpack.lastswe[b])
    snowpack.packsnowdensity[~b] = parameters['snow_dens_default']

    _apply_snowpack_temperature_instability_correction(snowpack, input_forcings, sec_in_ts)
        
    return updated_runoff, lastenergy


def run_snowclim_model(forcings_data, parameters):
    """
    Simulates snow accumulation, melting, sublimation, condensation, and energy balance
    over a given time period based on meteorological inputs.

    Parameters:
        forcings_data (dict): meteorological inputs.
        parameters (dict): Parameters required by the model.

    Returns:
        snow_model_instances (list): list with the results based on time.
    """
    # number of seconds in each time step
    sec_in_ts = parameters['hours_in_ts'] * const.HR_2_SECS
    coords = forcings_data['coords']
    size_lat = coords['lat'].size
    snow_model_instances = [None] * len(forcings_data['coords']['time'])
        
    for i, time_value in enumerate(forcings_data['coords']['time']):
        print(i)
        # loading necessary data to run the model
        input_forcings, snow_vars, previous_energy = _process_forcings_and_energy(i, forcings_data, parameters, snow_model_instances)

        # Calculate new precipitation components
        precip = _perform_precipitation_operations(input_forcings, parameters)

        # Reset to 0 snow at the specified time of year
        if i == 0 or (time_value[1] == parameters['snowoff_month'] and time_value[2] == parameters['snowoff_day']):
            snowpack = Snowpack(size_lat, parameters)
        
        snowfallcc  = _calculate_snow_temp_and_cold_content(precip.sfe, input_forcings, snowpack)
        snow_vars.CCsnowfall = snowfallcc.copy()

        # If there is snow on the ground, run the model
        exist_snow = (precip.sfe + snowpack.lastswe) > 0
        if np.sum(exist_snow) > 0:
            snow_vars.ExistSnow = exist_snow.copy()
            # Set snow surface temperature
            snow_vars.SnowTemp[exist_snow] = np.minimum(input_forcings['tdmean'] + parameters['Ts_add'],
                                                        0)[exist_snow]
            
            updated_runoff, lastenergy = _process_snowpack(
                 input_forcings, parameters, snow_vars, precip, snowpack, sec_in_ts, 
                 coords, time_value, previous_energy)

            # # Update outputs            
            snow_vars.Runoff = updated_runoff.copy()
            snow_vars.RaininSnow = snowpack.rain_in_snow.copy()
            snow_vars.Albedo[exist_snow] = snowpack.lastalbedo[exist_snow].copy()
            snow_vars.PackCC = snowpack.lastpackcc.copy()
            snow_vars.SnowDepth = snowpack.lastsnowdepth.copy()
            snow_vars.SnowWaterEq = snowpack.lastswe.copy()
            b = snowpack.lastswe > 0
            if np.any(b):
                snow_vars.SnowDensity[b] = snowpack.packsnowdensity[b].copy()
        else:
            snowpack.initialize_snowpack_base()
            
        snow_model_instances[i] = _prepare_outputs(snow_vars, precip)
        
    return snow_model_instances
