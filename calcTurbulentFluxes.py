"""
Calculate Turbulent Heat and Mass Fluxes Using Richardson Number Parameterization.
"""
import numpy as np

def calc_turbulent_fluxes(stability, windHt, z_0, tempHt, z_h, k, vs, lastsnowtemp, tavg, 
                          Ca, psfc, huss, E0_value, E0_app, E0_stable, sec_in_ts):

    """
    Calculate turbulent fluxes of heat and mass (evaporation/sublimation) using 
    Richardson number parameterization based on Essery et al. (2013).
    
    Parameters:
    stability: Boolean indicating whether stability parameterization is used
    windHt: Wind measurement height (m)
    z_0: Roughness length (m)
    tempHt: Temperature measurement height (m)
    z_h: Reference height for temperature profile (m)
    k: von Kármán constant (dimensionless)
    vs: Wind speed (m/s)
    lastsnowtemp: Snow surface temperature (C)
    tavg: Air temperature (C)
    Ca: Specific heat of air (kJ/kg/K)
    psfc: Surface pressure (hPa)
    huss: Specific humidity (kg/kg)
    E0_value: Windless exchange coefficient (W/m2/K)
    E0_app: Option for applying windless exchange (1 or 2)
    E0_stable: Stability option for windless exchange (1 or 2)
    sec_in_ts: Seconds in the current timestep
    
    Returns:
    H: Sensible heat flux (kJ/m2/timestep)
    E: Latent heat flux (kg/m2/timestep)
    EV: Energy flux due to evaporation or sublimation (kJ/m2/timestep)
    """
    
    # Constants
    R = 287  # gas constant for dry air (J K-1 kg-1)
    g = 9.80616  # gravitational acceleration (m/s^2)
    c = 5  # stability constant from Essery et al., (2013), table 6
    E0 = E0_value / 1000  # Convert E0 to kJ/m2/K/s
    
    # Convert air pressure from hPa to Pa
    psfcpa = psfc * 100
    
    # Calculate air density (kg/m^3)
    pa = psfcpa / (R * (tavg + 273.15))
    
    # Calculate vapor densities (kg/m^3)
    rhoa = huss
    rhos = calc_specific_humidity(lastsnowtemp, psfc)
    
    # Exchange coefficient for neutral conditions (CHN)
    CHN = k**2 / (np.log(windHt / z_0) * np.log(tempHt / z_h))
    
    if stability:
        # Calculate the bulk Richardson number
        Rib = (g * windHt * (tavg - lastsnowtemp)) / ((tavg + 273.15) * vs**2)
        
        # Calculate FH as a function of Rib
        FH = np.full_like(tavg, np.nan)
        # For unstable case
        FH[Rib < 0] = 1 - ((3 * c * Rib[Rib < 0]) / (1 + 3 * c**2 * CHN * (-Rib[Rib < 0] * windHt / z_0)**0.5))
        # For neutral case
        FH[Rib == 0] = 1
        # For stable case
        FH[Rib > 0] = (1 + ((2 * c * Rib[Rib > 0]) / (1 + Rib[Rib > 0])**0.5))**-1
        
        # Calculate exchange coefficient CH
        CH = FH * CHN
    else:
        CH = CHN
    
    # Latent heat of vaporization and sublimation
    LatHeatVap = calc_latent_heat_vap(lastsnowtemp)  # kJ/kg
    LatHeatSub = calc_latent_heat_sub(lastsnowtemp)  # kJ/kg
    
    # Windless exchange coefficient
    Ex = np.zeros_like(vs)
    if E0_stable == 1:
        Ex = E0
    elif E0_stable == 2:
        Ex[Rib > 0] = E0
    
    # Sensible heat flux (H)
    H = -(pa * Ca * CH * vs + Ex) * (lastsnowtemp - tavg)
    
    # Latent heat flux (E)
    if E0_app == 1:
        E = -(pa * CH * vs) * (rhos - rhoa)  # Mass flux kg/m2/s
    elif E0_app == 2:
        E = -(pa * CH * vs + Ex) * (rhos - rhoa)  # Mass flux kg/m2/s
    
    # Evaporation and sublimation energy flux (EV)
    Evap = E * LatHeatVap
    Esub = E * LatHeatSub
    EV = np.where(lastsnowtemp >= 0, Evap, Esub)
    
    # Convert from per second to per timestep
    H *= sec_in_ts  # kJ/m2/s
    E *= sec_in_ts  # kg/m2/s
    EV *= sec_in_ts  # kJ/m2/s
    
    # Avoid NaNs for zero wind speed
    H[vs == 0] = 0
    E[vs == 0] = 0
    EV[vs == 0] = 0
    
    return H, E, EV
