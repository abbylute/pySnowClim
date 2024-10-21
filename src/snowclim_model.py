import numpy as np
import pickle

import constants as const

from calcPhase import calculate_phase
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


def run_snowclim_model(forcings_data, parameters):
    """
    Simulates snow accumulation, melting, sublimation, condensation, and energy balance 
    over a given time period based on meteorological inputs.
    
    Parameters:
        lat (array-like): Array of latitudes for the grid points to model (1 x space).
        lrad (array-like): Downward longwave radiation (kJ/m²/hr) (time x space).
        tavg (array-like): Average air temperature (°C) (time x space).
        ppt (array-like): Precipitation (m) (time x space).
        solar (array-like): Downward shortwave radiation (kJ/m²/hr) (time x space).
        tdmean (array-like): Dewpoint temperature (°C) (time x space).
        vs (array-like): Wind speed (m/s) (time x space).
        relhum (array-like): Relative humidity (%) (time x space).
        psfc (array-like): Surface air pressure (hPa or mb) (time x space).
        huss (array-like): Specific humidity (kg/kg) (time x space).
        parameterfilename (str): Filepath to the parameter file containing model calibration data.
    
    Returns:
        tuple: Contains the following outputs (all time x space arrays unless otherwise noted):
            SnowMelt (array-like): Snow melt (m).
            SnowWaterEq (array-like): Snow water equivalent (m).
            SFE (array-like): Snowfall water equivalent (m).
            SnowDepth (array-like): Snow depth (mm).
            SnowDensity (array-like): Snowpack density (kg/m³).
            Sublimation (array-like): Snow sublimation (m).
            Condensation (array-like): Snow condensation (m).
            SnowTemp (array-like): Snow surface temperature (°C).
            MeltEnergy (array-like): Energy used for melting snow (kJ/m²/timestep).
            Energy (array-like): Net energy to the snowpack (kJ/m²/timestep).
            Albedo (array-like): Snow surface albedo.
            SnowYN (array-like): Snow cover binary (1 for snow, 0 for no snow).
            RaininSnow (array-like): Rain added to the snowpack (m).
            Runoff (array-like): Runoff from the snowpack (m).
            RefrozenWater (array-like): Liquid water refrozen in the snowpack (m).
            PackWater (array-like): Liquid water present in the snowpack (m).
            LW_down (array-like): Downward longwave radiation to the snow surface (kJ/m²/timestep).
            LW_up (array-like): Upward longwave radiation from the snow surface (kJ/m²/timestep).
            SW_down (array-like): Downward shortwave radiation to the snow surface (kJ/m²/timestep).
            SW_up (array-like): Upward shortwave radiation from the snow surface (kJ/m²/timestep).
            Q_latent (array-like): Latent heat flux (kJ/m²/timestep).
            Q_sensible (array-like): Sensible heat flux (kJ/m²/timestep).
            Q_precip (array-like): Precipitation heat flux (kJ/m²/timestep).
            PackCC (array-like): Snowpack cold content (kJ/m²/timestep).
            CCenergy (array-like): Cold content changes due to energy flux (kJ/m²/timestep).
            CCsnowfall (array-like): Cold content added by snowfall (kJ/m²/timestep).
    """

    cal = parameters['cal']

    ppt = forcings_data['ppt']
    #--- Allocate space ---
    outdim = ppt.shape

    SnowDepth = np.full(outdim, np.nan, dtype=np.float32)
    SnowWaterEq = np.full(outdim, np.nan, dtype=np.float32)
    SnowMelt = np.full(outdim, np.nan, dtype=np.float32)
    Sublimation = np.full(outdim, np.nan, dtype=np.float32)
    Condensation = np.full(outdim, np.nan, dtype=np.float32)
    Runoff = np.full(outdim, np.nan, dtype=np.float32)
    RaininSnow = np.full(outdim, np.nan, dtype=np.float32)
    RefrozenWater = np.full(outdim, np.nan, dtype=np.float32)
    PackWater = np.full(outdim, np.nan, dtype=np.float32)
    SnowYN = np.full(outdim, np.nan, dtype=np.float32)

    SnowTemp = np.full(outdim, np.nan, dtype=np.float32)
    Albedo = np.full(outdim, np.nan, dtype=np.float32)
    SnowDensity = np.full(outdim, np.nan, dtype=np.float32)

    Q_sensible = np.full(outdim, np.nan, dtype=np.float32)
    Q_latent = np.full(outdim, np.nan, dtype=np.float32)
    Q_precip = np.full(outdim, np.nan, dtype=np.float32)
    LW_up = np.full(outdim, np.nan, dtype=np.float32)
    LW_down = np.full(outdim, np.nan, dtype=np.float32)
    SW_up = np.full(outdim, np.nan, dtype=np.float32)
    SW_down = np.full(outdim, np.nan, dtype=np.float32)
    Energy = np.full(outdim, np.nan, dtype=np.float32)
    MeltEnergy = np.full(outdim, np.nan, dtype=np.float32)
    PackCC = np.full(outdim, np.nan, dtype=np.float32)
    CCsnowfall = np.full(outdim, np.nan, dtype=np.float32)
    CCenergy = np.full(outdim, np.nan, dtype=np.float32)

    #--- Converted Inputs ---
    # sec_in_ts = S['hours_in_ts'] * 3600
    sec_in_ts = parameters['hours_in_ts'] * const.MIN_2_SECS

    tavg_K = forcings_data['tavg'] + const.K_2_C
    passnow = calculate_phase(tavg_K, np.ones(tavg_K.shape), forcings_data['relhum'], 
                              False)
    
    R_m = ppt * (1 - passnow)
    SFE = ppt * passnow

    SFE[SFE < (0.0001 * parameters['hours_in_ts'])] = 0
    R_m[SFE < (0.0001 * parameters['hours_in_ts'])] = ppt[SFE < (0.0001 * parameters['hours_in_ts'])]

    newsnowdensity = calc_fresh_snow_density(forcings_data['tavg'])

    #--- For each time step ---
    for i in range(cal.shape[0]):        
        # Reset to 0 snow at the specified time of year
        if i == 0 or (cal[i, 1] == parameters['snowoff_month'] and cal[i, 2] == parameters['snowoff_day']):
            lastalbedo = np.ones(forcings_data['lat'].size) * parameters['ground_albedo']
            lastpacktemp = np.zeros(forcings_data['lat'].size)
            lastswe = np.zeros(forcings_data['lat'].size)
            lastsnowdepth = np.zeros(forcings_data['lat'].size)
            packsnowdensity = np.ones(forcings_data['lat'].size) * parameters['snow_dens_default']
            snowage = np.zeros(forcings_data['lat'].size)
            lastpackcc = np.zeros(forcings_data['lat'].size)
            lastpackwater = np.zeros(forcings_data['lat'].size)

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
        CCsnowfall[i, :] = snowfallcc

        # Update last pack temperature where there's snowfall
        if np.any(f):
            lastpacktemp[f] = lastpackcc[f] / (const.WATERDENS * const.CI * (lastswe[f] + newswe[f]))
        # lastpacktemp = np.where(f, 
        #                         lastpackcc / (const.WATERDENS * const.CI * (lastswe + newswe)), 
        #                         lastpacktemp)

        # If there is snow on the ground, run the model
        a = (newswe + lastswe) > 0
        if np.sum(a) > 0:
            SnowYN[i, :] = a

            # --- Set snow surface temperature ---
            lastsnowtemp = np.minimum(forcings_data['tdmean'][i, :] + parameters['Ts_add'], 0)
            SnowTemp[i, a] = lastsnowtemp[a]

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
            Runoff[i, :], lastpackwater = update_pack_water(a, 
                                                            lastpackwater, 
                                                            lastsnowdepth, 
                                                            parameters['lw_max'], 
                                                            Runoff[i, :], 
                                                            sec_in_ts)
            RaininSnow[i, a] = np.maximum(lastpackwater[a] - previouspackwater, 0)

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
            Albedo[i, a] = lastalbedo[a]

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
            Q_precip[i, :] = P

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
            Energy[i, :] = lastenergy

            # --- Apply cold content tax ---
            lastenergy = np.nanmean(Energy[max(1, (i - parameters['smooth_hr'] + 1)):i + 1, :], axis=0)
            tax = (lastpackcc - parameters['Tstart']) / parameters['Tadd'] * parameters['maxtax']
            tax = np.clip(tax, 0, parameters['maxtax'])  # limit tax to be >= 0 and <= maxtax
            n = lastenergy < 0
            if np.any(n):
                lastenergy[n] *= (1 - tax[n])

            # --- Distribute energy ---

            # 1. Energy goes to cold content first
            lastpackcc, lastenergy, CCenergy[i, :] = calc_energy_to_cc(lastpackcc, 
                                                                       lastenergy, 
                                                                       CCenergy[i, :])
            b = lastswe > 0
            if np.any(b):
                lastpacktemp[b] = lastpackcc[b] / (const.WATERDENS * const.CI * lastswe[b])

            # Apply temperature instability correction
            thres = sec_in_ts / const.MIN_2_SECS * 0.015  # 15mm for each hour in the time step
            f = np.where((lastswe < thres) & (lastswe > 0) & (lastpacktemp < forcings_data['tavg'][i, :]))
            lastpacktemp[f] = np.minimum(0, forcings_data['tavg'][i, f])
            lastpackcc[f] = const.WATERDENS * const.CI * lastswe[f] * lastpacktemp[f]

            # 2. Energy goes to refreezing second
            lastpackwater, lastswe, lastpackcc, packsnowdensity, RefrozenWater[i, :] = calc_energy_to_refreezing(lastpackwater,
                                                                                                                 lastswe,
                                                                                                                 lastpackcc,
                                                                                                                 lastsnowdepth,
                                                                                                                 RefrozenWater[i,:],
                                                                                                                 packsnowdensity)

            # 3. Energy goes to melt third
            SnowMelt[i, :], MeltEnergy[i, :], lastpackwater, lastswe, lastsnowdepth = calc_energy_to_melt(lastswe,
                                                                                                          lastsnowdepth,
                                                                                                          packsnowdensity,
                                                                                                          lastenergy,
                                                                                                          lastpackwater,
                                                                                                          SnowMelt[i, :],
                                                                                                          MeltEnergy[i, :])

            # Update water in snowpack
            Runoff[i, :], lastpackwater = update_pack_water(a, 
                                                            lastpackwater, 
                                                            lastsnowdepth, 
                                                            parameters['lw_max'], 
                                                            Runoff[i, :], 
                                                            sec_in_ts)
            PackWater[i, :] = lastpackwater

            # --- Sublimation ---
            a = lastsnowdepth > 0
            if np.any(a):
                Sublimation[i, a], Condensation[i, a], lastswe[a], lastsnowdepth[a], lastpackcc[a], packsnowdensity[a], \
                    lastpackwater[a] = calc_sublimation(E[a], 
                                                       lastswe[a], 
                                                       lastsnowdepth[a], 
                                                       packsnowdensity[a], 
                                                       lastsnowtemp[a],
                                                       lastpackcc[a], 
                                                       parameters['snow_dens_default'], 
                                                       Sublimation[i, a], 
                                                       Condensation[i, a],
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
            PackCC[i, :] = lastpackcc
            SnowDepth[i, :] = lastsnowdepth
            SnowWaterEq[i, :] = lastswe
            b = lastswe > 0
            if np.any(b):
                SnowDensity[i, b] = packsnowdensity[b]

            SW_down[i, :] = Sdn
            SW_up[i, :] = Sup
            LW_down[i, :] = Ldn
            LW_up[i, :] = Lt
            Q_latent[i, :] = EV
            Q_sensible[i, :] = H
        else:
            # Initialize arrays with default values
            lastalbedo = np.ones(forcings_data['lat'].size, dtype=np.float32) * parameters['ground_albedo']
            lastswe = np.zeros(forcings_data['lat'].size, dtype=np.float32)
            lastsnowdepth = np.zeros(forcings_data['lat'].size, dtype=np.float32)
            packsnowdensity = np.ones(forcings_data['lat'].size, dtype=np.float32) * parameters['snow_dens_default']
            lastpackwater = np.zeros(forcings_data['lat'].size, dtype=np.float32)
            lastpackcc = np.zeros(forcings_data['lat'].size, dtype=np.float32)


    #--- Prepare outputs ---
    SnowWaterEq = SnowWaterEq * const.WATERDENS
    SFE = SFE * const.WATERDENS
    SnowMelt = SnowMelt * const.WATERDENS
    Sublimation = Sublimation * const.WATERDENS
    Condensation = Condensation * const.WATERDENS
    SnowDepth = SnowDepth * 1000
    SnowDensity = SnowDensity
    Runoff = Runoff * const.WATERDENS
    RaininSnow = RaininSnow * const.WATERDENS
    RefrozenWater = RefrozenWater * const.WATERDENS
    PackWater = PackWater * const.WATERDENS
    Albedo = Albedo
    SnowTemp = SnowTemp
    PackCC = PackCC
    CCsnowfall = CCsnowfall
    CCenergy = CCenergy

    Energy = Energy
    MeltEnergy = MeltEnergy
    LW_up = LW_up
    SW_up = SW_up
    LW_down = LW_down
    SW_down = SW_down
    Q_latent = Q_latent
    Q_sensible = Q_sensible
    Q_precip = Q_precip
    

    return (SnowMelt, SnowWaterEq, SFE, SnowDepth, SnowDensity, Sublimation, 
            Condensation, SnowTemp, MeltEnergy, Energy, Albedo, SnowYN, RaininSnow, 
            Runoff, RefrozenWater, PackWater, LW_down, LW_up, SW_down, SW_up, 
            Q_latent, Q_sensible, Q_precip, PackCC, CCenergy, CCsnowfall)
