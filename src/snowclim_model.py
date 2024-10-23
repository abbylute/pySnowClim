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

def perform_precipitation_operations(forcings_data, parameters):
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
    - sec_in_ts: float
        Number of seconds in each time step.
    - R_m: np.ndarray
        Rainfall amount after phase calculation.
    - SFE: np.ndarray
        Snowfall equivalent after phase calculation.
    - newsnowdensity: np.ndarray
        Fresh snow density based on the average temperature.
    """

    # Calculate phase (snow or rain fraction)
    passnow = calc_phase(forcings_data['tavg'], 
                         forcings_data['relhum'], 
                         False)  # False means not binary (returns fraction)

    # Separate rain and snow components of precipitation
    R_m = forcings_data['ppt'] * (1 - passnow)
    SFE = forcings_data['ppt'] * passnow

    # Threshold for snowfall equivalent and adjust rain accordingly
    threshold = 0.0001 * parameters['hours_in_ts']
    SFE[SFE < threshold] = 0
    R_m[SFE < threshold] = forcings_data['ppt'][SFE < threshold]

    # Calculate fresh snow density based on temperature
    newsnowdensity = calc_fresh_snow_density(forcings_data['tavg'])

    return R_m, SFE, newsnowdensity


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

    cal = parameters['cal']

    outdim = forcings_data['ppt'].shape
    model_vars = SnowModelVariables(forcings_data['ppt'].shape)

    # SnowDepth = np.full(outdim, np.nan, dtype=np.float32)
    #SnowWaterEq = np.full(outdim, np.nan, dtype=np.float32)
    # SnowMelt = np.full(outdim, np.nan, dtype=np.float32)
    # Sublimation = np.full(outdim, np.nan, dtype=np.float32)
    #Condensation = np.full(outdim, np.nan, dtype=np.float32)
    #Runoff = np.full(outdim, np.nan, dtype=np.float32)
    #RaininSnow = np.full(outdim, np.nan, dtype=np.float32)
    # RefrozenWater = np.full(outdim, np.nan, dtype=np.float32)
    # PackWater = np.full(outdim, np.nan, dtype=np.float32)
    # SnowYN = np.full(outdim, np.nan, dtype=np.float32)

    # SnowTemp = np.full(outdim, np.nan, dtype=np.float32)
    # Albedo = np.full(outdim, np.nan, dtype=np.float32)
    # SnowDensity = np.full(outdim, np.nan, dtype=np.float32)

    # Q_sensible = np.full(outdim, np.nan, dtype=np.float32)
    # Q_latent = np.full(outdim, np.nan, dtype=np.float32)
    # Q_precip = np.full(outdim, np.nan, dtype=np.float32)
    # LW_up = np.full(outdim, np.nan, dtype=np.float32)
    # LW_down = np.full(outdim, np.nan, dtype=np.float32)
    # SW_up = np.full(outdim, np.nan, dtype=np.float32)
    # SW_down = np.full(outdim, np.nan, dtype=np.float32)
    # Energy = np.full(outdim, np.nan, dtype=np.float32)
    # MeltEnergy = np.full(outdim, np.nan, dtype=np.float32)
    # PackCC = np.full(outdim, np.nan, dtype=np.float32)
    # CCsnowfall = np.full(outdim, np.nan, dtype=np.float32)
    # CCenergy = np.full(outdim, np.nan, dtype=np.float32)

    # number of seconds in each time step
    sec_in_ts = parameters['hours_in_ts'] * const.MIN_2_SECS
    R_m, SFE, newsnowdensity = perform_precipitation_operations(forcings_data, parameters)

    #--- For each time step ---
    for i in range(cal.shape[0]):        
        # Reset to 0 snow at the specified time of year
        if i == 0 or (cal[i, 1] == parameters['snowoff_month'] and cal[i, 2] == parameters['snowoff_day']):
            (lastalbedo, lastswe, lastsnowdepth, packsnowdensity, lastpackcc, 
             lastpackwater) = initialize_snowpack_variables(forcings_data['lat'].size,
                                                            parameters)
            lastpacktemp = np.zeros(forcings_data['lat'].size, dtype=np.float32)
            snowage = np.zeros(forcings_data['lat'].size, dtype=np.float32)
                                                                        
        # --- new mass inputs ---
        newswe = SFE[i, :]
        newrain = R_m[i, :]
        newsnowdepth = SFE[i, :] * const.WATERDENS / newsnowdensity[i, :]
        newsnowdens = newsnowdensity[i, :]

        # --- Calculate snow temperature and cold contents ---
        newsnowtemp = np.zeros(forcings_data['lat'].size, dtype=np.float32)  # Initialize array for new snow temperatures
        f = newswe > 0  # Boolean mask where there's new snow

        # Update new snow temperature where newswe > 0
        newsnowtemp = np.where(f, np.minimum(0, forcings_data['tdmean'][i]), newsnowtemp)
        # newsnowtemp[f] = np.minimum(0, forcings_data['tdmean'][i, f])

        # Calculate cold content of snowfall
        snowfallcc = const.WATERDENS * const.CI * newswe * newsnowtemp
        lastpackcc += snowfallcc

        # Store snowfall cold content for this timestep
        model_vars.CCsnowfall[i, :] = snowfallcc

        # Update last pack temperature where there's snowfall
        if np.any(f):
            lastpacktemp[f] = lastpackcc[f] / (const.WATERDENS * const.CI * (lastswe[f] + newswe[f]))
        # lastpacktemp = np.where(f, 
        #                         lastpackcc / (const.WATERDENS * const.CI * (lastswe + newswe)), 
        #                         lastpacktemp)

        # If there is snow on the ground, run the model
        a = (newswe + lastswe) > 0
        if np.sum(a) > 0:
            model_vars.SnowYN[i, :] = a

            # --- Set snow surface temperature ---
            lastsnowtemp = np.minimum(forcings_data['tdmean'][i, :] + parameters['Ts_add'], 0)
            model_vars.SnowTemp[i, a] = lastsnowtemp[a]

            # --- Update snowpack after new snowfall ---
            packsnowdensity[a] = calc_snow_density_after_snow(lastswe[a], 
                                                                   newswe[a], 
                                                                   packsnowdensity[a], 
                                                                   newsnowdens[a])
            lastswe += newswe
            lastsnowdepth += newsnowdepth

            # --- Calculate pack density after compaction ---
            packsnowdensity[a] = calc_snow_density(lastswe[a], 
                                                        lastpacktemp[a], 
                                                        packsnowdensity[a], 
                                                        sec_in_ts)
            lastsnowdepth[a] = lastswe[a] * const.WATERDENS / packsnowdensity[a]

            # --- Update snowpack liquid water content ---
            previouspackwater = lastpackwater[a]
            lastpackwater[a] += newrain[a]
            model_vars.Runoff[i, :], lastpackwater = update_pack_water(a, 
                                                            lastpackwater, 
                                                            lastsnowdepth, 
                                                            parameters['lw_max'], 
                                                            model_vars.Runoff[i, :], 
                                                            sec_in_ts)
            model_vars.RaininSnow[i, a] = np.maximum(lastpackwater[a] - previouspackwater, 0)

            # --- Calculate albedo ---
            lastalbedo, snowage = calc_albedo(parameters['albedo_option'], 
                                                   parameters['ground_albedo'], 
                                                   parameters['max_albedo'], 
                                                   lastalbedo, 
                                                   newsnowdepth,
                                                   lastsnowdepth, 
                                                   newswe, 
                                                   lastswe, 
                                                   lastsnowtemp, 
                                                   forcings_data['lat'], 
                                                   cal[i, 2], 
                                                   cal[i, 3],
                                                   snowage, 
                                                   lastpackcc, 
                                                   sec_in_ts)
            model_vars.Albedo[i, a] = lastalbedo[a]

            # --- Calculate turbulent heat fluxes (kJ/m2/timestep) ---
            H = np.zeros(forcings_data['lat'].size, dtype=np.float32)
            E = np.zeros(forcings_data['lat'].size, dtype=np.float32)
            EV = np.zeros(forcings_data['lat'].size, dtype=np.float32)
            H[a], E[a], EV[a] = calc_turbulent_fluxes(parameters['stability'], 
                                                        parameters['windHt'], 
                                                        parameters['z_0'], 
                                                        parameters['tempHt'], 
                                                        parameters['z_h'], 
                                                        forcings_data['vs'][i, a],
                                                        lastsnowtemp[a], 
                                                        forcings_data['tavg'][i, a], 
                                                        forcings_data['psfc'][i, a], 
                                                        forcings_data['huss'][i, a], 
                                                        parameters['E0_value'],
                                                        parameters['E0_app'], 
                                                        parameters['E0_stable'], 
                                                        sec_in_ts)

            # --- Rain heat flux into snowpack (kJ/m2/timestep) ---
            P = np.zeros(lastsnowtemp.size, dtype=np.float32)
            P[a] = const.CW * const.WATERDENS * np.maximum(0, forcings_data['tdmean'][i, a]) * newrain[a]
            model_vars.Q_precip[i, :] = P

            # --- Net downward solar flux at surface (kJ/m2/timestep) ---
            Sup = np.zeros(forcings_data['lat'].size, dtype=np.float32)
            Sup[a] = forcings_data['solar'][i, a] * lastalbedo[a]
            Sdn = np.zeros(forcings_data['lat'].size, dtype=np.float32)
            Sdn[a] = forcings_data['solar'][i, a]

            # --- Longwave flux up from snow surface (kJ/m2/timestep) ---
            Ldn = np.zeros(forcings_data['lat'].size, dtype=np.float32)
            Ldn[a] = forcings_data['lrad'][i, a]
            Lt = np.zeros(forcings_data['lat'].size, dtype=np.float32)
            Lt[a] = calc_longwave(parameters['snow_emis'], 
                                  lastsnowtemp[a], 
                                  forcings_data['lrad'][i, a], 
                                  sec_in_ts)

            # --- Ground heat flux (kJ/m2/timestep) ---
            Gf = np.zeros(forcings_data['lat'].size, dtype=np.float32)
            Gf[a] = parameters['G'] * sec_in_ts

            # --- Downward net energy flux into snow surface (kJ/m2/timestep) ---
            lastenergy = Sdn - Sup + Ldn - Lt + H + EV + Gf + P
            model_vars.Energy[i, :] = lastenergy

            # --- Apply cold content tax ---
            lastenergy = np.nanmean(model_vars.Energy[max(1, (i - parameters['smooth_hr'] + 1)):i + 1, :], axis=0)
            tax = (lastpackcc - parameters['Tstart']) / parameters['Tadd'] * parameters['maxtax']
            tax = np.clip(tax, 0, parameters['maxtax'])  # limit tax to be >= 0 and <= maxtax
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
                lastpacktemp[b] = lastpackcc[b] / (const.WATERDENS * const.CI * lastswe[b])

            # Apply temperature instability correction
            thres = sec_in_ts / const.MIN_2_SECS * 0.015  # 15mm for each hour in the time step
            f = np.where((lastswe < thres) & (lastswe > 0) & (lastpacktemp < forcings_data['tavg'][i, :]))
            lastpacktemp[f] = np.minimum(0, forcings_data['tavg'][i, f])
            lastpackcc[f] = const.WATERDENS * const.CI * lastswe[f] * lastpacktemp[f]

            # 2. Energy goes to refreezing second
            lastpackwater, lastswe, lastpackcc, packsnowdensity, model_vars.RefrozenWater[i, :] = calc_energy_to_refreezing(lastpackwater,
                                                                                                                 lastswe,
                                                                                                                 lastpackcc,
                                                                                                                 lastsnowdepth,
                                                                                                                 model_vars.RefrozenWater[i,:],
                                                                                                                 packsnowdensity)

            # 3. Energy goes to melt third
            model_vars.SnowMelt[i, :], model_vars.MeltEnergy[i, :], lastpackwater, lastswe, lastsnowdepth = calc_energy_to_melt(lastswe,
                                                                                                          lastsnowdepth,
                                                                                                          packsnowdensity,
                                                                                                          lastenergy,
                                                                                                          lastpackwater,
                                                                                                          model_vars.SnowMelt[i, :],
                                                                                                          model_vars.MeltEnergy[i, :])

            # Update water in snowpack
            model_vars.Runoff[i, :], lastpackwater = update_pack_water(a, 
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
            f = np.where((lastswe < thres) & (lastswe > 0) & (lastpacktemp < forcings_data['tavg'][i, :]))
            lastpacktemp[f] = np.minimum(0, forcings_data['tavg'][i, f])
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
             lastpackwater) = initialize_snowpack_variables(forcings_data['lat'].size,
                                                            parameters)


    #--- Prepare outputs ---
    model_vars.SnowWaterEq = model_vars.SnowWaterEq * const.WATERDENS
    SFE = SFE * const.WATERDENS
    model_vars.SnowMelt = model_vars.SnowMelt * const.WATERDENS
    model_vars.Sublimation = model_vars.Sublimation * const.WATERDENS
    model_vars.Condensation = model_vars.Condensation * const.WATERDENS
    model_vars.SnowDepth = model_vars.SnowDepth * 1000
    model_vars.SnowDensity = model_vars.SnowDensity
    model_vars.Runoff = model_vars.Runoff * const.WATERDENS
    model_vars.RaininSnow = model_vars.RaininSnow * const.WATERDENS
    model_vars.RefrozenWater = model_vars.RefrozenWater * const.WATERDENS
    model_vars.PackWater = model_vars.PackWater * const.WATERDENS
    model_vars.Albedo = model_vars.Albedo
    model_vars.SnowTemp = model_vars.SnowTemp
    
    return model_vars
