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


def initialize_snowpack_variables(n_lat, parameters):
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
    model_vars = SnowModelVariables(forcings_data['forcings']['ppt'].shape)

    # number of seconds in each time step
    sec_in_ts = parameters['hours_in_ts'] * const.MIN_2_SECS
    rainfall, SnowfallWaterEq, newsnowdensity = _perform_precipitation_operations(
        forcings_data['forcings'], parameters)

    coords = forcings_data['coords']
    size_lat = coords['lat'].size
    
    # --- For each time step ---
    for i, time_value in enumerate(forcings_data['coords']['time']):
        input_forcings = {key: value[i, :] for key, value in forcings_data['forcings'].items()}

        # Reset to 0 snow at the specified time of year
        if i == 0 or (time_value[1] == parameters['snowoff_month'] and time_value[2] == parameters['snowoff_day']):
            (lastalbedo, lastswe, lastsnowdepth, packsnowdensity, lastpackcc,
             lastpackwater) = initialize_snowpack_variables(size_lat,
                                                            parameters)
            lastpacktemp = np.zeros(size_lat, dtype=np.float32)
            snowage = np.zeros(size_lat, dtype=np.float32)

        # --- new mass inputs ---
        newswe = SnowfallWaterEq[i, :]
        newrain = rainfall[i, :]
        newsnowdepth = SnowfallWaterEq[i, :] * const.WATERDENS / newsnowdensity[i, :]
        newsnowdens = newsnowdensity[i, :]

        # --- Calculate snow temperature and cold contents ---
        has_new_snow = newswe > 0

        # Update new snow temperature where newswe > 0
        newsnowtemp = np.where(has_new_snow, np.minimum(0, input_forcings['tdmean']), 0)

        # Calculate cold content of snowfall
        snowfallcc = const.WATERDENS * const.CI * newswe * newsnowtemp
        lastpackcc += snowfallcc

        # Store snowfall cold content for this timestep
        model_vars.CCsnowfall[i, :] = snowfallcc

        # Update last pack temperature where there's snowfall
        if np.any(has_new_snow):
            lastpacktemp[has_new_snow] = lastpackcc[has_new_snow] / \
                (const.WATERDENS * const.CI *
                 (lastswe[has_new_snow] + newswe[has_new_snow]))

        # If there is snow on the ground, run the model
        exist_snow = (newswe + lastswe) > 0
        if np.sum(exist_snow) > 0:
            model_vars.SnowYN[i, :] = exist_snow

            # --- Set snow surface temperature ---
            lastsnowtemp = np.minimum(input_forcings['tdmean'] + parameters['Ts_add'], 0)
            model_vars.SnowTemp[i, exist_snow] = lastsnowtemp[exist_snow]

            # --- Update snowpack after new snowfall ---
            packsnowdensity[exist_snow] = calc_snow_density_after_snow(lastswe[exist_snow],
                                                                       newswe[exist_snow],
                                                                       packsnowdensity[exist_snow],
                                                                       newsnowdens[exist_snow])
            lastswe += newswe
            lastsnowdepth += newsnowdepth

            # --- Calculate pack density after compaction ---
            packsnowdensity[exist_snow] = calc_snow_density(lastswe[exist_snow],
                                                            lastpacktemp[exist_snow],
                                                            packsnowdensity[exist_snow],
                                                            sec_in_ts)
            lastsnowdepth[exist_snow] = lastswe[exist_snow] * \
                const.WATERDENS / packsnowdensity[exist_snow]

            # --- Update snowpack liquid water content ---
            previouspackwater = lastpackwater[exist_snow]
            lastpackwater[exist_snow] += newrain[exist_snow]
            model_vars.Runoff[i, :], lastpackwater = update_pack_water(exist_snow,
                                                                       lastpackwater,
                                                                       lastsnowdepth,
                                                                       parameters['lw_max'],
                                                                       model_vars.Runoff[i, :],
                                                                       sec_in_ts)
            model_vars.RaininSnow[i, exist_snow] = np.maximum(
                lastpackwater[exist_snow] - previouspackwater, 0)

            # --- Calculate albedo ---
            lastalbedo, snowage = calc_albedo(parameters,
                                              lastalbedo,
                                              newsnowdepth,
                                              lastsnowdepth,
                                              newswe,
                                              lastswe,
                                              lastsnowtemp,
                                              coords['lat'].ravel(),
                                              time_value[1],
                                              time_value[2],
                                              snowage,
                                              lastpackcc,
                                              sec_in_ts)
            model_vars.Albedo[i, exist_snow] = lastalbedo[exist_snow]

            # --- Calculate turbulent heat fluxes (kJ/m2/timestep) ---
            H = np.zeros(size_lat, dtype=np.float32)
            E = np.zeros(size_lat, dtype=np.float32)
            EV = np.zeros(size_lat, dtype=np.float32)
            has_snow_and_wind = exist_snow & (input_forcings['vs'] > 0)
            H[has_snow_and_wind], E[has_snow_and_wind], EV[has_snow_and_wind] = calc_turbulent_fluxes(
                parameters,
                input_forcings['vs'][has_snow_and_wind],
                lastsnowtemp[has_snow_and_wind],
                input_forcings['tavg'][has_snow_and_wind],
                input_forcings['psfc'][has_snow_and_wind],
                input_forcings['huss'][has_snow_and_wind],
                sec_in_ts)

            # --- Rain heat flux into snowpack (kJ/m2/timestep) ---
            P = np.zeros(lastsnowtemp.size, dtype=np.float32)
            P[exist_snow] = const.CW * const.WATERDENS * \
                np.maximum(0, input_forcings['tdmean'][exist_snow]) * newrain[exist_snow]
            model_vars.Q_precip[i, :] = P

            # --- Net downward solar flux at surface (kJ/m2/timestep) ---
            Sup = np.zeros(size_lat, dtype=np.float32)
            Sup[exist_snow] = input_forcings['solar'][exist_snow] * lastalbedo[exist_snow]
            Sdn = np.zeros(size_lat, dtype=np.float32)
            Sdn[exist_snow] = input_forcings['solar'][exist_snow]

            # --- Longwave flux up from snow surface (kJ/m2/timestep) ---
            Ldn = np.zeros(size_lat, dtype=np.float32)
            Ldn[exist_snow] = input_forcings['lrad'][exist_snow]
            Lt = np.zeros(size_lat, dtype=np.float32)
            Lt[exist_snow] = calc_longwave(parameters['snow_emis'],
                                           lastsnowtemp[exist_snow],
                                           input_forcings['lrad'][exist_snow],
                                           sec_in_ts)

            # --- Ground heat flux (kJ/m2/timestep) ---
            Gf = np.where(exist_snow, parameters['G'] * sec_in_ts, 0)

            # --- Downward net energy flux into snow surface (kJ/m2/timestep) ---
            lastenergy = Sdn - Sup + Ldn - Lt + H + EV + Gf + P
            model_vars.Energy[i, :] = lastenergy

            # --- Apply cold content tax ---
            lastenergy = np.nanmean(model_vars.Energy[max(
                1, (i - parameters['smooth_hr'] + 1)):i + 1, :], axis=0)
            tax = (lastpackcc - parameters['Tstart']) / \
                parameters['Tadd'] * parameters['maxtax']
            # limit tax to be >= 0 and <= maxtax
            tax = np.clip(tax, 0, parameters['maxtax'])
            n = lastenergy < 0
            if np.any(n):
                lastenergy[n] *= (1 - tax[n])

            # --- Distribute energy ---

            # 1. Energy goes to cold content first
            lastpackcc, lastenergy, model_vars.CCenergy[i, :] = calc_energy_to_cc(lastpackcc,
                                                                                  lastenergy,
                                                                                  model_vars.CCenergy[i, :])
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
            lastpackwater, lastswe, lastpackcc, packsnowdensity, model_vars.RefrozenWater[i, :] = calc_energy_to_refreezing(lastpackwater,
                                                                                                                            lastswe,
                                                                                                                            lastpackcc,
                                                                                                                            lastsnowdepth,
                                                                                                                            model_vars.RefrozenWater[
                                                                                                                                i, :],
                                                                                                                            packsnowdensity)

            # 3. Energy goes to melt third
            model_vars.SnowMelt[i, :], model_vars.MeltEnergy[i, :], lastpackwater, lastswe, lastsnowdepth = calc_energy_to_melt(lastswe,
                                                                                                                                lastsnowdepth,
                                                                                                                                packsnowdensity,
                                                                                                                                lastenergy,
                                                                                                                                lastpackwater,
                                                                                                                                model_vars.SnowMelt[
                                                                                                                                    i, :],
                                                                                                                                model_vars.MeltEnergy[i, :])

            # Update water in snowpack
            model_vars.Runoff[i, :], lastpackwater = update_pack_water(exist_snow,
                                                                       lastpackwater,
                                                                       lastsnowdepth,
                                                                       parameters['lw_max'],
                                                                       model_vars.Runoff[i, :],
                                                                       sec_in_ts)
            model_vars.PackWater[i, :] = lastpackwater

            # --- Sublimation ---
            a = lastsnowdepth > 0
            if np.any(a):
                model_vars.Sublimation[i, a], model_vars.Condensation[i, a], lastswe[a], lastsnowdepth[a], lastpackcc[a], packsnowdensity[a], \
                    lastpackwater[a] = calc_sublimation(E[a],
                                                        lastswe[a],
                                                        lastsnowdepth[a],
                                                        packsnowdensity[a],
                                                        lastsnowtemp[a],
                                                        lastpackcc[a],
                                                        parameters['snow_dens_default'],
                                                        model_vars.Sublimation[i, a],
                                                        model_vars.Condensation[i, a],
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
            model_vars.PackCC[i, :] = lastpackcc
            model_vars.SnowDepth[i, :] = lastsnowdepth
            model_vars.SnowWaterEq[i, :] = lastswe
            b = lastswe > 0
            if np.any(b):
                model_vars.SnowDensity[i, b] = packsnowdensity[b]

            model_vars.SW_down[i, :] = Sdn
            model_vars.SW_up[i, :] = Sup
            model_vars.LW_down[i, :] = Ldn
            model_vars.LW_up[i, :] = Lt
            model_vars.Q_latent[i, :] = EV
            model_vars.Q_sensible[i, :] = H
        else:
            # Initialize arrays with default values
            (lastalbedo, lastswe, lastsnowdepth, packsnowdensity, lastpackcc,
             lastpackwater) = initialize_snowpack_variables(size_lat, parameters)

    # --- Prepare outputs ---
    model_vars = _prepare_outputs(model_vars, SnowfallWaterEq)

    return model_vars
