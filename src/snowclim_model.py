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


def _prepare_outputs(model_vars, SnowfallWaterEq):
    """
    Prepare outputs by scaling model variables with appropriate constants.

    Args:
    -----
    - model_vars: object
        An object containing model variables such as SnowWaterEq, SnowDepth, etc.
    - SnowfallWaterEq: float
        The snowfall water equivalent to be scaled.

    Returns:
    --------
    Updated model_vars with scaled values for output.
    """
    # Scale model variables by WATERDENS or other units where applicable
    model_vars.SnowWaterEq *= const.WATERDENS
    model_vars.SnowfallWaterEq = SnowfallWaterEq * const.WATERDENS
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

    return rainfall, SnowfallWaterEq, newsnowdensity


def _initialize_snowpack_variables(n_lat, parameters):
    """
    Initialize variables related to snowpack and albedo.

    Args:
    -----
    - nlat: integer
        Number of latitude points.
    - parameters: dict
        A dictionary of model parameters, including 'ground_albedo' and 'snow_dens_default'.

    Returns:
    --------
    - lastalbedo: np.ndarray
        Albedo values initialized based on the ground albedo parameter.
    - lastswe: np.ndarray
        Last snow water equivalent initialized to zeros.
    - lastsnowdepth: np.ndarray
        Last snow depth initialized to zeros.
    - packsnowdensity: np.ndarray
        Snow density initialized based on the default parameter.
    - lastpackcc: np.ndarray
        Last pack cold content initialized to zeros.
    - lastpackwater: np.ndarray
        Last pack water content initialized to zeros.
    """
    # Initialize arrays
    lastalbedo = np.ones(n_lat, dtype=np.float32) * parameters['ground_albedo']
    lastswe = np.zeros(n_lat, dtype=np.float32)
    lastsnowdepth = np.zeros(n_lat, dtype=np.float32)
    packsnowdensity = np.ones(n_lat, dtype=np.float32) * parameters['snow_dens_default']
    lastpackcc = np.zeros(n_lat, dtype=np.float32)
    lastpackwater = np.zeros(n_lat, dtype=np.float32)

    return (lastalbedo, lastswe, lastsnowdepth, packsnowdensity, lastpackcc, lastpackwater)


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

    # Smooth energy values if enough timesteps have passed
    smoothed_energy = None
    if index > parameters['smooth_hr']:
        smoothed_energy = np.full((parameters['smooth_hr'], size_lat), np.nan)
        for l in range(parameters['smooth_hr']):
            smoothed_energy[l, :] = np.maximum(1, snow_model_instances[index - l - 1].Energy)

    return input_forcings, snow_vars, smoothed_energy


def _calculate_snow_temp_and_cold_content(newswe, input_forcings, lastpackcc, 
                                          lastswe, lastpacktemp):
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
    updated_lastpackcc = lastpackcc + snowfallcc

    # Update last pack temperature where there is snowfall
    if np.any(has_new_snow):
        lastpacktemp[has_new_snow] = updated_lastpackcc[has_new_snow] / \
            (const.WATERDENS * const.CI * (lastswe[has_new_snow] + newswe[has_new_snow]))

    return snowfallcc, lastpacktemp


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
    has_snow_and_wind = exist_snow & (input_forcings['vs'] > 0)
    H, E, EV = calc_turbulent_fluxes(
        parameters, input_forcings['vs'][has_snow_and_wind], lastsnowtemp[has_snow_and_wind],
        input_forcings['tavg'][has_snow_and_wind], input_forcings['psfc'][has_snow_and_wind],
        input_forcings['huss'][has_snow_and_wind], sec_in_ts
    )
    energy_var['Q_sensible'][has_snow_and_wind] = H
    energy_var['E'][has_snow_and_wind] = E
    energy_var['Q_latent'][has_snow_and_wind] = E

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

    # Copy smoothed energy for modification
    previous_energy[parameters['smooth_hr']-1,:] = lastenergy
    smoothed_energy = np.nanmean(previous_energy, axis=0)
    # Apply tax where energy is negative
    negative_energy = smoothed_energy < 0
    if np.any(negative_energy):
        smoothed_energy[negative_energy] *= (1 - tax[negative_energy])

    return smoothed_energy


def run_snowclim_model(forcings_data, parameters):
    """
    Simulates snow accumulation, melting, sublimation, condensation, and energy balance
    over a given time period based on meteorological inputs.

    Parameters:
        lat (array-like): Array of latitudes for the grid points to model (1 x space).
        lrad (array-like): Downward longwave radiation (kJ/m²/hr) (time x space).

    Returns:
        model_vars (SnowModelVariables class):
    """
    # number of seconds in each time step
    sec_in_ts = parameters['hours_in_ts'] * const.MIN_2_SECS

    coords = forcings_data['coords']
    size_lat = coords['lat'].size
    snow_model_instances = [None] * len(forcings_data['coords']['time'])

    # --- For each time step ---
    for i, time_value in enumerate(forcings_data['coords']['time']):
        # loading necessary data to run the model
        input_forcings, snow_vars, previous_energy = _process_forcings_and_energy(
            i, forcings_data, parameters, snow_model_instances)

        # Calculate new precipitation components
        newrain, newswe, newsnowdens = _perform_precipitation_operations(input_forcings, parameters)
        newsnowdepth = newswe * const.WATERDENS / newsnowdens

        # Reset to 0 snow at the specified time of year
        if i == 0 or (time_value[1] == parameters['snowoff_month'] and time_value[2] == parameters['snowoff_day']):
            (lastalbedo, lastswe, lastsnowdepth, packsnowdensity, lastpackcc,
             lastpackwater) = _initialize_snowpack_variables(size_lat, parameters)
            lastpacktemp = np.zeros(size_lat, dtype=np.float32)
            snowage = np.zeros(size_lat, dtype=np.float32)
        
        snowfallcc, lastpacktemp = _calculate_snow_temp_and_cold_content(
            newswe, input_forcings, lastpackcc, lastswe, lastpacktemp)
        lastpackcc += snowfallcc
        snow_vars.CCsnowfall = snowfallcc.copy()

        # If there is snow on the ground, run the model
        exist_snow = (newswe + lastswe) > 0
        if np.sum(exist_snow) > 0:
            snow_vars.ExistSnow = exist_snow.copy()

            # --- Set snow surface temperature ---
            lastsnowtemp = np.minimum(input_forcings['tdmean'] + parameters['Ts_add'], 0)
            snow_vars.SnowTemp[exist_snow] = lastsnowtemp[exist_snow].copy()

            # --- Update snowpack after new snowfall ---
            packsnowdensity[exist_snow] = calc_snow_density_after_snow(
                lastswe[exist_snow], newswe[exist_snow], packsnowdensity[exist_snow],
                newsnowdens[exist_snow])
            
            lastswe += newswe
            lastsnowdepth += newsnowdepth

            # --- Calculate pack density after compaction ---
            packsnowdensity[exist_snow] = calc_snow_density(
                lastswe[exist_snow], lastpacktemp[exist_snow], packsnowdensity[exist_snow],
                sec_in_ts)
            lastsnowdepth[exist_snow] = lastswe[exist_snow] * \
                const.WATERDENS / packsnowdensity[exist_snow]

            # --- Update snowpack liquid water content ---
            previouspackwater = lastpackwater[exist_snow]
            lastpackwater[exist_snow] += newrain[exist_snow]
            snow_vars.Runoff, lastpackwater = update_pack_water(
                exist_snow, lastpackwater, lastsnowdepth, parameters['lw_max'],
                snow_vars.Runoff, sec_in_ts)
            snow_vars.RaininSnow[exist_snow] = np.maximum(
                lastpackwater[exist_snow] - previouspackwater, 0)

            # --- Calculate albedo ---
            lastalbedo, snowage = calc_albedo(
                parameters, lastalbedo, newsnowdepth, lastsnowdepth, newswe, lastswe,
                lastsnowtemp, coords['lat'].ravel(), time_value[1], time_value[2],
                snowage, lastpackcc, sec_in_ts)
            snow_vars.Albedo[exist_snow] = lastalbedo[exist_snow].copy()

            energy_var = _calculate_energy_fluxes(exist_snow, parameters, input_forcings,
                                                    lastsnowtemp, newrain, lastalbedo, 
                                                    sec_in_ts)
            
            # --- Downward net energy flux into snow surface (kJ/m2/timestep) ---
            lastenergy = energy_var['SW_down'] - energy_var['SW_up']  
            lastenergy += energy_var['LW_down'] - energy_var['LW_up'] 
            lastenergy += energy_var['Q_sensible'] + energy_var['Q_latent']  
            lastenergy += energy_var['Gf'] + energy_var['Q_precip']
            snow_vars.Energy = lastenergy.copy()

            # copying values from the dict to the object
            for key, value in energy_var.items():
                if hasattr(snow_vars, key):
                    setattr(snow_vars, key, value)

            taxed_last_energy = _apply_cold_content_tax(
                lastpackcc, parameters, previous_energy, lastenergy)         

            # --- Distribute energy ---
            # 1. Energy goes to cold content first
            lastpackcc, taxed_last_energy, snow_vars.CCenergy = calc_energy_to_cc(
                lastpackcc, taxed_last_energy, snow_vars.CCenergy)
            
            b = lastswe > 0
            if np.any(b):
                lastpacktemp[b] = lastpackcc[b] / \
                    (const.WATERDENS * const.CI * lastswe[b])

            # Apply temperature instability correction
            thres = sec_in_ts / const.MIN_2_SECS * 0.015  # 15mm for each hour in the time step
            f = np.where((lastswe < thres) & (lastswe > 0) &
                         (lastpacktemp < input_forcings['tavg']))
            lastpacktemp[f] = np.minimum(0, input_forcings['tavg'][f])
            lastpackcc[f] = const.WATERDENS * const.CI * lastswe[f] * lastpacktemp[f]

            # 2. Energy goes to refreezing second
            lastpackwater, lastswe, lastpackcc, packsnowdensity, snow_vars.RefrozenWater = calc_energy_to_refreezing(
                lastpackwater, lastswe, lastpackcc, lastsnowdepth, snow_vars.RefrozenWater, packsnowdensity)

            # 3. Energy goes to melt third
            snow_vars.SnowMelt, snow_vars.MeltEnergy, lastpackwater, lastswe, lastsnowdepth = calc_energy_to_melt(
                lastswe, lastsnowdepth, packsnowdensity, taxed_last_energy, lastpackwater, 
                snow_vars.SnowMelt, snow_vars.MeltEnergy)

            # Update water in snowpack
            snow_vars.Runoff, lastpackwater = update_pack_water(
                exist_snow, lastpackwater, lastsnowdepth, parameters['lw_max'],
                snow_vars.Runoff, sec_in_ts)
            snow_vars.PackWater = lastpackwater.copy()

            # --- Sublimation ---
            a = lastsnowdepth > 0
            if np.any(a):
                snow_vars.Sublimation[a], snow_vars.Condensation[a], lastswe[a], lastsnowdepth[a], lastpackcc[a], packsnowdensity[a], \
                    lastpackwater[a] = calc_sublimation(
                        energy_var['E'][a], lastswe[a], lastsnowdepth[a], packsnowdensity[a],
                        lastsnowtemp[a], lastpackcc[a], parameters['snow_dens_default'],
                        snow_vars.Sublimation[a], snow_vars.Condensation[a],
                        lastpackwater[a])

            # Update snow
            b = lastswe > 0
            lastpackwater[~b] = 0
            lastalbedo[~b] = parameters['ground_albedo']
            snowage[~b] = 0
            lastpacktemp[~b] = 0
            lastpackcc[~b] = 0
            lastsnowdepth[~b] = 0
            lastpacktemp[b] = lastpackcc[b] / (const.WATERDENS * const.CI * lastswe[b])
            packsnowdensity[~b] = parameters['snow_dens_default']

            # Apply temperature instability correction
            thres = sec_in_ts / const.MIN_2_SECS * 0.015  # 15mm for each hour in the time step
            f = np.where((lastswe < thres) & (lastswe > 0) &
                         (lastpacktemp < input_forcings['tavg']))
            lastpacktemp[f] = np.minimum(0, input_forcings['tavg'][f])
            lastpackcc[f] = const.WATERDENS * const.CI * lastswe[f] * lastpacktemp[f]

            # Update outputs
            snow_vars.PackCC = lastpackcc.copy()
            snow_vars.SnowDepth = lastsnowdepth.copy()
            snow_vars.SnowWaterEq = lastswe.copy()
            b = lastswe > 0
            if np.any(b):
                snow_vars.SnowDensity[b] = packsnowdensity[b]

        else:
            # Initialize arrays with default values
            (lastalbedo, lastswe, lastsnowdepth, packsnowdensity, lastpackcc,
             lastpackwater) = _initialize_snowpack_variables(size_lat, parameters)
            
        snow_model_instances[i] = _prepare_outputs(snow_vars, newswe)

    return snow_model_instances
