import numpy as np
import pickle

#--- Constants ---
WATERDENS = 1000  # density of water (kg/m3)
LatHeatFreez = 333.3  # latent heat of freezing (kJ/kg)
Ci = 2.117  # heat Capacity of Snow (kJ/kg/C)
Cw = 4.2  # heat Capacity of Water (kJ/kg/C)
Ca = 1.005  # heat Capacity of Air (kJ/kg/C)
k = 0.41  # von Karman's constant

def snowclim_model(lat, lrad, tavg, ppt, solar, tdmean, vs, relhum, psfc, huss, parameterfilename):
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


    print('Running SnowClim model...')

    
    #--- Parameters ---
    with open(parameterfilename, 'rb') as f:
        S = pickle.load(f)
    cal = S['cal']


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
    sec_in_ts = S['hours_in_ts'] * 3600

    passnow = calcPhase(tavg + 273.15, np.ones(tavg.shape), relhum, False)
    R_m = ppt * (1 - passnow)
    SFE = ppt * passnow

    SFE[SFE < (0.0001 * S['hours_in_ts'])] = 0
    R_m[SFE < (0.0001 * S['hours_in_ts'])] = ppt[SFE < (0.0001 * S['hours_in_ts'])]

    newsnowdensity = calcFreshSnowDensity(tavg)

    #--- For each time step ---
    for i in range(cal.shape[0]):
        
        # Reset to 0 snow at the specified time of year
        if i == 1 or (cal[i, 1] == S['snowoff_month'] and cal[i, 2] == S['snowoff_day']):
            lastalbedo = np.ones(lat.shape) * S['ground_albedo']
            lastpacktemp = np.zeros(lat.shape)
            lastswe = np.zeros(lat.shape)
            lastsnowdepth = np.zeros(lat.shape)
            packsnowdensity = np.ones(lat.shape) * S['snow_dens_default']
            snowage = np.zeros(lat.shape)
            lastpackcc = np.zeros(lat.shape)
            lastpackwater = np.zeros(lat.shape)

            # --- new mass inputs ---
            newswe = SFE[i, :]
            newrain = R_m[i, :]
            newsnowdepth = SFE[i, :] * WaterDens / newsnowdensity[i, :]
            newsnowdens = newsnowdensity[i, :]

            # --- Calculate snow temperature and cold contents ---
            newsnowtemp = np.zeros(len(lat), dtype=np.float32)  # Initialize array for new snow temperatures
            f = newswe > 0  # Boolean mask where there's new snow

            # Update new snow temperature where newswe > 0
            newsnowtemp[f] = np.minimum(0, tdmean[i, f])

            # Calculate cold content of snowfall
            snowfallcc = WaterDens * Ci * newswe * newsnowtemp
            lastpackcc += snowfallcc

            # Store snowfall cold content for this timestep
            CCsnowfall[i, :] = snowfallcc

            # Update last pack temperature where there's snowfall
            lastpacktemp[f] = lastpackcc[f] / (WaterDens * Ci * (lastswe[f] + newswe[f]))


            # If there is snow on the ground, run the model
            a = (newswe + lastswe) > 0
            if np.sum(a) > 0:
                SnowYN[i, :] = a

                # --- Set snow surface temperature ---
                lastsnowtemp = np.minimum(tdmean[i, :] + S.Ts_add, 0)
                SnowTemp[i, a] = lastsnowtemp[a]

                # --- Update snowpack after new snowfall ---
                packsnowdensity[a] = calcSnowDensityAfterSnow(lastswe[a], newswe[a], packsnowdensity[a], newsnowdens[a])
                lastswe += newswe
                lastsnowdepth += newsnowdepth

                # --- Calculate pack density after compaction ---
                packsnowdensity[a] = calcSnowDensity(lastswe[a], lastpacktemp[a], packsnowdensity[a], WaterDens, sec_in_ts)
                lastsnowdepth[a] = lastswe[a] * WaterDens / packsnowdensity[a]

                # --- Update snowpack liquid water content ---
                previouspackwater = lastpackwater[a]
                lastpackwater[a] += newrain[a]
                Runoff[i, :], lastpackwater = updatePackWater(a, lastpackwater, lastsnowdepth, S.lw_max, Runoff[i, :], sec_in_ts)
                RaininSnow[i, a] = np.maximum(lastpackwater[a] - previouspackwater, 0)

                # --- Calculate albedo ---
                lastalbedo, snowage = calcAlbedo(S.albedo_option, S.ground_albedo, S.max_albedo, lastalbedo, newsnowdepth,
                                                lastsnowdepth, newswe, lastswe, lastsnowtemp, WaterDens, lat, cal[i, 2], cal[i, 3],
                                                snowage, lastpackcc, sec_in_ts)
                Albedo[i, a] = lastalbedo[a]

                # --- Calculate turbulent heat fluxes (kJ/m2/timestep) ---
                H = np.zeros(len(lat), dtype=np.float32)
                E = np.zeros(len(lat), dtype=np.float32)
                EV = np.zeros(len(lat), dtype=np.float32)
                H[a], E[a], EV[a] = calcTurbulentFluxes(S.stability, S.windHt, S.z_0, S.tempHt, S.z_h, k, vs[i, a],
                                                        lastsnowtemp[a], tavg[i, a], Ca, psfc[i, a], huss[i, a], S.E0_value,
                                                        S.E0_app, S.E0_stable, sec_in_ts)

                # --- Rain heat flux into snowpack (kJ/m2/timestep) ---
                P = np.zeros(len(lastsnowtemp), dtype=np.float32)
                P[a] = Cw * WaterDens * np.maximum(0, tdmean[i, a]) * newrain[a]
                Q_precip[i, :] = P

                # --- Net downward solar flux at surface (kJ/m2/timestep) ---
                Sup = np.zeros(len(lat), dtype=np.float32)
                Sup[a] = solar[i, a] * lastalbedo[a]
                Sdn = np.zeros(len(lat), dtype=np.float32)
                Sdn[a] = solar[i, a]

                # --- Longwave flux up from snow surface (kJ/m2/timestep) ---
                Ldn = np.zeros(len(lat), dtype=np.float32)
                Ldn[a] = lrad[i, a]
                Lt = np.zeros(len(lat), dtype=np.float32)
                Lt[a] = calcLongwave(S.snow_emis, lastsnowtemp[a], lrad[i, a], sec_in_ts)

                # --- Ground heat flux (kJ/m2/timestep) ---
                Gf = np.zeros(len(lat), dtype=np.float32)
                Gf[a] = S.G * sec_in_ts

                # --- Downward net energy flux into snow surface (kJ/m2/timestep) ---
                lastenergy = Sdn - Sup + Ldn - Lt + H + EV + Gf + P
                Energy[i, :] = lastenergy

                # --- Apply cold content tax ---
                lastenergy = np.nanmean(Energy[max(1, (i - S.smooth_hr + 1)):i + 1, :], axis=0)
                tax = (lastpackcc - S.Tstart) / S.Tadd * S.maxtax
                tax = np.clip(tax, 0, S.maxtax)  # limit tax to be >= 0 and <= maxtax
                n = lastenergy < 0
                lastenergy[n] *= (1 - tax[n])

                # --- Distribute energy ---

                # 1. Energy goes to cold content first
                lastpackcc, lastenergy, CCenergy[i, :] = calcEnergyToCC(lastpackcc, lastenergy, CCenergy[i, :])
                b = lastswe > 0
                lastpacktemp[b] = lastpackcc[b] / (WaterDens * Ci * lastswe[b])

                # Apply temperature instability correction
                thres = sec_in_ts / 3600 * 0.015  # 15mm for each hour in the time step
                f = np.where((lastswe < thres) & (lastswe > 0) & (lastpacktemp < tavg[i, :]))
                lastpacktemp[f] = np.minimum(0, tavg[i, f])
                lastpackcc[f] = WaterDens * Ci * lastswe[f] * lastpacktemp[f]

                # 2. Energy goes to refreezing second
                lastpackwater, lastswe, lastpackcc, packsnowdensity, RefrozenWater[i, :] = calcEnergyToRefreezing(lastpackwater,
                                                                                                                lastswe,
                                                                                                                lastpackcc,
                                                                                                                lastsnowdepth,
                                                                                                                WaterDens,
                                                                                                                LatHeatFreez,
                                                                                                                RefrozenWater[i,
                                                                                                                            :],
                                                                                                                packsnowdensity)

                # 3. Energy goes to melt third
                SnowMelt[i, :], MeltEnergy[i, :], lastpackwater, lastswe, lastsnowdepth = calcEnergyToMelt(lastswe,
                                                                                                        lastsnowdepth,
                                                                                                        packsnowdensity,
                                                                                                        lastenergy,
                                                                                                        lastpackwater,
                                                                                                        LatHeatFreez, WaterDens,
                                                                                                        SnowMelt[i, :],
                                                                                                        MeltEnergy[i, :])

                # Update water in snowpack
                Runoff[i, :], lastpackwater = updatePackWater(a, lastpackwater, lastsnowdepth, S.lw_max, Runoff[i, :], sec_in_ts)
                PackWater[i, :] = lastpackwater

                # --- Sublimation ---
                a = lastsnowdepth > 0
                Sublimation[i, a], Condensation[i, a], lastswe[a], lastsnowdepth[a], lastpackcc[a], packsnowdensity[a], \
                    lastpackwater[a] = calcSublimation(E[a], lastswe[a], lastsnowdepth[a], packsnowdensity[a], lastsnowtemp[a],
                                                    lastpackcc[a], S.snow_dens_default, Sublimation[i, a], Condensation[i, a],
                                                    lastpackwater[a], WaterDens)

                # Update snow
                b = lastswe > 0
                lastpackwater[~b] = 0
                lastalbedo[~b] = S.ground_albedo
                snowage[~b] = 0
                lastpacktemp[~b] = 0
                lastpackcc[~b] = 0
                lastsnowdepth[~b] = 0
                lastpacktemp[b] = lastpackcc[b] / (WaterDens * Ci * lastswe[b])
                packsnowdensity[~b] = S.snow_dens_default

                # Apply temperature instability correction
                thres = sec_in_ts / 3600 * 0.015  # 15mm for each hour in the time step
                f = np.where((lastswe < thres) & (lastswe > 0) & (lastpacktemp < tavg[i, :]))
                lastpacktemp[f] = np.minimum(0, tavg[i, f])
                lastpackcc[f] = WaterDens * Ci * lastswe[f] * lastpacktemp[f]

                # Update outputs
                PackCC[i, :] = lastpackcc
                SnowDepth[i, :] = lastsnowdepth
                SnowWaterEq[i, :] = lastswe
                b = lastswe > 0
                SnowDensity[i, b] = packsnowdensity[b]

                SW_down[i, :] = Sdn
                SW_up[i, :] = Sup
                LW_down[i, :] = Ldn
                LW_up[i, :] = Lt
                Q_latent[i, :] = EV
                Q_sensible[i, :] = H
            else:
                # Initialize arrays with default values
                lastalbedo = np.ones(len(lat), dtype=np.float32) * S.ground_albedo
                lastswe = np.zeros(len(lat), dtype=np.float32)
                lastsnowdepth = np.zeros(len(lat), dtype=np.float32)
                packsnowdensity = np.ones(len(lat), dtype=np.float32) * S.snow_dens_default
                lastpackwater = np.zeros(len(lat), dtype=np.float32)
                lastpackcc = np.zeros(len(lat), dtype=np.float32)


    #--- Prepare outputs ---
    SnowWaterEq = SnowWaterEq * WATERDENS
    SFE = SFE * WATERDENS
    SnowMelt = SnowMelt * WATERDENS
    Sublimation = Sublimation * WATERDENS
    Condensation = Condensation * WATERDENS
    SnowDepth = SnowDepth * 1000
    SnowDensity = SnowDensity
    Runoff = Runoff * WATERDENS
    RaininSnow = RaininSnow * WATERDENS
    RefrozenWater = RefrozenWater * WATERDENS
    PackWater = PackWater * WATERDENS
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

# Define the additional functions used such as calcPhase, calcFreshSnowDensity, etc.
